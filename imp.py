import requests
import numpy as np
from datetime import datetime
from sgp4.api import Satrec
import time
import streamlit as st
import plotly.graph_objects as go
import random
from pyproj import Transformer

# NASA API Key 
NASA_API_KEY = "5xnePBAC4QbxoYf0hFOnZx3eZjebfsQI1cZiAiPW"

# 🔹 Function to fetch real-time TLE data using NORAD ID
def fetch_tle(norad_id):
    url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=tle"
    response = requests.get(url)

    if response.status_code != 200 or not response.text.strip():
        st.error(f"Error fetching TLE data: {response.status_code}")
        return []

    tle_lines = response.text.strip().split("\n")
    
    if len(tle_lines) < 3:
        st.error("TLE data not found.")
        return []

    return tle_lines[:3]  # Return only 3 lines (name + TLE 1 + TLE 2)

# 🔹 Convert datetime to Julian Date
def datetime_to_julian(dt):
    jd = 367 * dt.year - (7 * (dt.year + ((dt.month + 9) // 12)) // 4) + (275 * dt.month // 9) + dt.day + 1721013.5
    fr = (dt.hour * 3600 + dt.minute * 60 + dt.second) / 86400.0
    return jd, fr

# 🔹 Propagate satellite's position
def propagate_satellite_position(satellite, jd, fr):
    return satellite.sgp4(jd, fr)

# 🔹 Convert ECI coordinates to Lat, Lon, Alt
def eci_to_latlonalt(position):
    transformer = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)
    lon, lat, alt = transformer.transform(position[0] * 1000, position[1] * 1000, position[2] * 1000)
    return lat, lon, alt / 1000  # Convert meters to km

# 🔹 Generate realistic debris positions near the satellite
def generate_debris_positions(satellite_position, num_debris=50):
    debris_positions = []
    for _ in range(num_debris):
        debris_positions.append(np.array([
            satellite_position[0] + random.uniform(-300, 300),  # Keeping debris closer
            satellite_position[1] + random.uniform(-300, 300),
            satellite_position[2] + random.uniform(-300, 300)
        ]))
    return np.array(debris_positions)

# 🔹 Calculate distance and risk levels
def calculate_risk_levels(satellite_position, debris_positions):
    distances = np.linalg.norm(debris_positions - satellite_position, axis=1)

    risk_levels = np.where(
        distances < 50, "High Risk",  # Less than 50 km = High Risk
        np.where(distances < 200, "Medium Risk", "Low Risk")  # 50-200 km = Medium, else Low
    )

    return debris_positions, distances, risk_levels

# 🔹 Track and visualize satellite & debris in real-time
def track_satellite(norad_id):
    tle_lines = fetch_tle(norad_id)
    if not tle_lines:
        return
    
    satellite = Satrec.twoline2rv(tle_lines[1], tle_lines[2])

    # Streamlit UI
    st.title("Real-Time Satellite & Debris Tracker ")
    st.write(f"**Tracking:** {tle_lines[0]}")
    st.write(f"**NORAD ID:** {norad_id}")

    satellite_positions = []
    
    while True:
        now = datetime.utcnow()
        jd, fr = datetime_to_julian(now)
        e, r, v = propagate_satellite_position(satellite, jd, fr)
        if e != 0:
            st.error("Error in satellite propagation")
            return
        
        satellite_position = np.array(r)
        lat, lon, alt = eci_to_latlonalt(satellite_position)
        debris_positions = generate_debris_positions(satellite_position, num_debris=50)
        debris_positions, distances, risk_levels = calculate_risk_levels(satellite_position, debris_positions)
        satellite_positions.append(satellite_position)

        # 🚨 Alerts
        if any(distances < 50):
            st.error(" Collision Warning: High-risk debris detected within 50 km!")
        if alt < 300:
            st.warning("Altitude Alert: Satellite is below 300 km!")

        # 3D Visualization
        fig = go.Figure()

        # Plot satellite path (Orbit Trail Effect)
        fig.add_trace(go.Scatter3d(
            x=[pos[0] for pos in satellite_positions],
            y=[pos[1] for pos in satellite_positions],
            z=[pos[2] for pos in satellite_positions],
            mode='lines+markers',
            name='Satellite Path',
            marker=dict(size=4, color='blue'),
            line=dict(width=2, color='deepskyblue')
        ))

        # Plot debris with risk levels (Ensure High-Risk debris is visible)
        risk_color_map = {"High Risk": "red", "Medium Risk": "orange", "Low Risk": "green"}
        risk_size_map = {"High Risk": 10, "Medium Risk": 6, "Low Risk": 4}

        for risk_level in ["High Risk", "Medium Risk", "Low Risk"]:
            indices = np.where(risk_levels == risk_level)[0]
            if len(indices) > 0:
                fig.add_trace(go.Scatter3d(
                    x=debris_positions[indices, 0],
                    y=debris_positions[indices, 1],
                    z=debris_positions[indices, 2],
                    mode='markers',
                    name=risk_level,
                    marker=dict(size=risk_size_map[risk_level], color=risk_color_map[risk_level], opacity=1.0)
                ))

        fig.update_layout(scene=dict(
            xaxis_title='X (km)',
            yaxis_title='Y (km)',
            zaxis_title='Z (km)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),  # Better 3D view
        ), title="Satellite and Space Debris Positions")

        st.plotly_chart(fig, use_container_width=True)

        # Display satellite position details
        st.subheader(" Satellite Position")
        st.write(f"**Latitude:** {lat:.2f}°")
        st.write(f"**Longitude:** {lon:.2f}°")
        st.write(f"**Altitude:** {alt:.2f} km")

        time.sleep(5)  # Reduced update time for smoother tracking

# **Run the Streamlit App**
if __name__ == "__main__":
    st.sidebar.title("Satellite Selection")
    norad_id = st.sidebar.text_input("Enter NORAD ID:", "20580")  # Default is ISS
    
    if st.sidebar.button("Track Satellite"):
        track_satellite(norad_id)
