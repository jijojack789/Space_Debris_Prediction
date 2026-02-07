import requests
import numpy as np
import torch
import torch.nn as nn
import os
import time
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from sgp4.api import Satrec
from pyproj import Transformer

# LSTM Model Definition
class SpaceDebrisLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, output_size=3):
        super(SpaceDebrisLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# Load Model
model = SpaceDebrisLSTM()
if os.path.exists("space_debris_lstm.pth"):
    model.load_state_dict(torch.load("space_debris_lstm.pth"))
    model.eval()

# Fetch TLE Data
def fetch_tle(norad_id):
    url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=tle"
    response = requests.get(url)
    return response.text.strip().split("\n")[:3] if response.status_code == 200 else []

# Convert Date to Julian
def datetime_to_julian(dt):
    jd = 367 * dt.year - (7 * (dt.year + (dt.month + 9) // 12) // 4) + (275 * dt.month // 9) + dt.day + 1721013.5
    fr = (dt.hour * 3600 + dt.minute * 60 + dt.second) / 86400.0
    return jd, fr

# Convert ECI to Lat, Lon, Alt
def eci_to_latlonalt(position):
    transformer = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)
    lon, lat, alt = transformer.transform(position[0] * 1000, position[1] * 1000, position[2] * 1000)
    return lat, lon, alt / 1000  # Convert meters to km

# Generate Debris
def generate_debris_positions(satellite_position, num_debris=30):
    return np.array([
        satellite_position + np.random.uniform(-250, 250, 3)
        for _ in range(num_debris)
    ])

# Calculate Distance & Risk
def calculate_risk_levels(satellite_position, debris_positions):
    distances = np.linalg.norm(debris_positions - satellite_position, axis=1)
    risk_levels = np.where(distances < 50, "High Risk", np.where(distances < 200, "Medium Risk", "Low Risk"))
    return distances, risk_levels

# Predict Future Position
def predict_future_position(model, position):
    with torch.no_grad():
        input_data = torch.tensor(position, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return model(input_data).squeeze().numpy()

# Track & Visualize Satellite and Debris
def track_satellite(norad_id):
    tle_lines = fetch_tle(norad_id)
    if not tle_lines:
        return
    
    satellite = Satrec.twoline2rv(tle_lines[1], tle_lines[2])
    st.title(" Space Debris Tracking & Collision Prediction")
    st.write(f"**Satellite:** {tle_lines[0]} | **NORAD ID:** {norad_id}")

    satellite_path = []  # Store last 30 positions

    while True:
        now = datetime.utcnow()
        jd, fr = datetime_to_julian(now)
        e, r, v = satellite.sgp4(jd, fr)
        if e != 0:
            st.error("Error in satellite propagation")
            return

        # Satellite & Debris Positions
        satellite_position = np.array(r)
        lat, lon, alt = eci_to_latlonalt(satellite_position)
        debris_positions = generate_debris_positions(satellite_position)
        distances, risk_levels = calculate_risk_levels(satellite_position, debris_positions)

        # Predict Future Positions
        future_satellite_position = predict_future_position(model, satellite_position)
        future_debris_positions = np.array([predict_future_position(model, debris) for debris in debris_positions])

        # Convert Future Positions to Lat, Lon, Alt
        future_lat, future_lon, future_alt = eci_to_latlonalt(future_satellite_position)

        # Track Satellite Path
        satellite_path.append(satellite_position)
        if len(satellite_path) > 30:
            satellite_path.pop(0)  # Keep last 30 points for smooth path

        # Identify Collision Risk Debris
        high_risk_indices = np.where(distances < 50)[0]
        collision_debris = high_risk_indices.tolist()

        # Alerts
        if collision_debris:
            st.error(f" Warning: {len(collision_debris)} debris objects are at high risk!")

        if alt < 300:
            st.warning(" Altitude Alert: Satellite is below 300 km!")

        # Display Future Prediction in Text
        st.subheader("Future Position Prediction")
        st.write(f"**Current Satellite Position:** Lat {lat:.4f}, Lon {lon:.4f}, Alt {alt:.2f} km")
        st.write(f"**Predicted Future Position:** Lat {future_lat:.4f}, Lon {future_lon:.4f}, Alt {future_alt:.2f} km")

        # 3D Visualization
        fig = go.Figure()

        # Satellite Path (Thicker Line)
        fig.add_trace(go.Scatter3d(
            x=[p[0] for p in satellite_path],
            y=[p[1] for p in satellite_path],
            z=[p[2] for p in satellite_path],
            mode='lines',
            line=dict(color='blue', width=6),  # Thick line
            name='Satellite Path'
        ))

        # Debris Markers with Risk Levels
        risk_colors = {"High Risk": "red", "Medium Risk": "orange", "Low Risk": "rgba(0,255,0,0.5)"}
        for i, (pos, risk, dist) in enumerate(zip(debris_positions, risk_levels, distances)):
            fig.add_trace(go.Scatter3d(
                x=[pos[0]], y=[pos[1]], z=[pos[2]],
                mode='markers+text' if risk == "High Risk" else 'markers',
                marker=dict(size=10 if risk == "High Risk" else 5, color=risk_colors[risk]),
                text=f"Debris {i+1} ({risk})" if risk == "High Risk" else "",
                name=f"Debris {i+1}"
            ))

        fig.update_layout(
            scene=dict(xaxis_title='X (km)', yaxis_title='Y (km)', zaxis_title='Z (km)'),
            title="Satellite & Debris Collision Prediction",
            margin=dict(l=0, r=0, b=0, t=40),
            legend=dict(x=0, y=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Display Debris Table
        debris_data = pd.DataFrame({
            "Debris #": np.arange(1, len(debris_positions) + 1),
            "Lat": [eci_to_latlonalt(p)[0] for p in debris_positions],
            "Lon": [eci_to_latlonalt(p)[1] for p in debris_positions],
            "Altitude (km)": [eci_to_latlonalt(p)[2] for p in debris_positions],
            "Future Lat": [eci_to_latlonalt(p)[0] for p in future_debris_positions],
            "Future Lon": [eci_to_latlonalt(p)[1] for p in future_debris_positions],
            "Future Alt": [eci_to_latlonalt(p)[2] for p in future_debris_positions],
            "Distance (km)": distances,
            "Risk Level": risk_levels
        })
        st.write(debris_data)

        time.sleep(3)

if __name__ == "__main__":
    norad_id = st.sidebar.text_input("Enter NORAD ID:", "20580")
    if st.sidebar.button("Track Satellite"):
        track_satellite(norad_id)
