import streamlit as st
import pandas as pd
import joblib
import random
import time
import folium
from streamlit_folium import st_folium
from geopy.distance import geodesic
from datetime import datetime
from utils import send_sms_alert  # Ensure the alert function is always imported

# Load model and scaler
scaler = joblib.load('models/scaler.pkl')
model = joblib.load('models/xgb_drone_classifier.pkl')

# Streamlit page configuration
st.set_page_config(page_title="Drone Activity Classifier", layout="wide")
st.title("🚁 Targeted Drone Activity Classifier")

# Sidebar settings
st.sidebar.header("🔧 Simulated Drone Parameters")
st.sidebar.markdown("""
Use the sliders to simulate various drone parameters. This allows for a realistic real-time simulation of the drone's behavior.
""")

# Simulate drone input
def simulate_drone_data():
    lat = 28.6139 + random.uniform(-0.01, 0.01)
    lon = 77.2090 + random.uniform(-0.01, 0.01)
    altitude = random.uniform(5, 100)
    speed = random.uniform(0, 25)
    distance_to_restricted = random.uniform(1, 200)
    return lat, lon, altitude, speed, distance_to_restricted

# Save state
if "counter" not in st.session_state:
    st.session_state.counter = 0
st.session_state.counter += 1

# Simulate drone data and add delay for real-time simulation
lat, lon, altitude, speed, distance = simulate_drone_data()

# Display simulated input in the sidebar
st.sidebar.write(f"📍 Latitude: {lat:.5f}")
st.sidebar.write(f"📍 Longitude: {lon:.5f}")
st.sidebar.write(f"🛫 Altitude: {altitude:.2f} m")
st.sidebar.write(f"💨 Speed: {speed:.2f} m/s")
st.sidebar.write(f"🚫 Distance to Restricted Zone: {distance:.2f} m")

# Feature Engineering
df = pd.DataFrame([[lat, lon, altitude, speed, distance]], 
                  columns=["lat", "lon", "altitude", "speed", "distance_to_restricted"])

df["speed_altitude_ratio"] = df["speed"] / (df["altitude"] + 1e-6)
df["proximity_score"] = 1 / (df["distance_to_restricted"] + 1e-6)
df["altitude_proximity_ratio"] = df["altitude"] / (df["distance_to_restricted"] + 1e-6)
df["speed_squared"] = df["speed"] ** 2
df["altitude_squared"] = df["altitude"] ** 2

features = ["lat", "lon", "altitude", "speed", "distance_to_restricted",
            "speed_altitude_ratio", "proximity_score", "altitude_proximity_ratio",
            "speed_squared", "altitude_squared"]

scaled_input = scaler.transform(df[features])
prediction = model.predict(scaled_input)
status = "Suspicious" if prediction[0] == 1 else "Normal"

# Improved visualization for prediction
st.subheader("🔍 **Prediction Status**")
status_color = "red" if status == "Suspicious" else "green"
st.markdown(f"""
    <div style="padding: 20px; background-color: {status_color}; color: white; font-size: 24px; border-radius: 10px; text-align: center;">
        **{status}**
    </div>
""", unsafe_allow_html=True)

# Display time of prediction
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.write(f"🕒 Last updated at: {current_time}")

# Map visualization with better styling
st.subheader("📍 Drone Location")
m = folium.Map(location=[lat, lon], zoom_start=13)
folium.Marker(
    location=[lat, lon],
    popup=f"Drone Status: {status}",
    icon=folium.Icon(color=status_color)
).add_to(m)
st_folium(m, width=700, height=500)

# SMS Alert with time interval control
if status == "Suspicious":
    if "last_alert_time" not in st.session_state:
        st.session_state.last_alert_time = datetime.now()

    time_since_last_alert = (datetime.now() - st.session_state.last_alert_time).seconds
    if time_since_last_alert > 300:  # 300 seconds = 5 minutes
        send_sms_alert(f"🚨 Suspicious Drone Detected!\nSpeed: {speed:.2f} m/s\nAltitude: {altitude:.2f} m")
        st.session_state.last_alert_time = datetime.now()  # Update the last alert time

# Auto refresh with a delay for real-time simulation
st.write("🔄 Refreshing in 25 seconds...")
time.sleep(25)  # Adjust this delay to control how often the page refreshes
st.experimental_rerun()
