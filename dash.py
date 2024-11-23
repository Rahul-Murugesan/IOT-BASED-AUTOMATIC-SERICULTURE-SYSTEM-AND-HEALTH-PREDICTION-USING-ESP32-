import streamlit as st
import numpy as np
import pandas as pd
import pyrebase
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Firebase Configuration
firebase_config = {
    "apiKey": "AIzaSyCZE61YtZWt_lMaZiOq6ANFvLCyrNWYVEQ",
    "authDomain": "sanket-e9152.firebaseapp.com",
    "databaseURL": "https://iotpro-ea2f4-default-rtdb.asia-southeast1.firebasedatabase.app",
    "projectId": "sanket-e9152",
    "storageBucket": "sanket-e9152.appspot.com",
    "messagingSenderId": "47809698488",
    "appId": "1:47809698488:web:753c4c7fe364de610e324d"
}

# Initialize Firebase
firebase = pyrebase.initialize_app(firebase_config)
database = firebase.database()

# Load the dataset
file_path = 'sericulture_dataset.csv'

try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    st.error("File not found. Please ensure 'sericulture_dataset.csv' is in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the dataset: {e}")
    st.stop()

# Check required columns
required_columns = ['Temperature (°C)', 'Humidity (%)', 'Light Intensity (Lux)', 
                    'Feeding Quality', 'Cocoon Yield (kg)']
if not all(column in data.columns for column in required_columns):
    st.error(f"The dataset is missing one or more required columns: {', '.join(required_columns)}")
    st.stop()

# Encode 'Health Status' if present
if 'Health Status' in data.columns:
    label_encoder = LabelEncoder()
    data['Health Status'] = label_encoder.fit_transform(data['Health Status'])

# Define features and target variable
features = ['Temperature (°C)', 'Humidity (%)', 'Light Intensity (Lux)', 'Feeding Quality']
target = 'Cocoon Yield (kg)'

X = data[features]
y = data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit App
st.title("Sericulture Monitoring and Cocoon Yield Prediction")

# Sidebar for feeding quality input
st.sidebar.header("Input Parameters")
feeding_quality = st.sidebar.selectbox("Feeding Quality (1 to 5)", [1, 2, 3, 4, 5])

# Function to fetch data from Firebase
def fetch_firebase_data():
    try:
        # Fetch data from Firebase
        data = database.get()
        if data.val():
            # Extract data for DHT_11
            dht_data = data.val().get("DHT_11", {})
            temperature = float(dht_data.get("Temperature", 0.0))
            humidity = float(dht_data.get("Humidity", 0.0))
            
            # Extract data for LDR sensor
            ldr_data = data.val().get("LDR", {})
            light_intensity = float(ldr_data.get("LightIntensity", 0.0))
            
            # Extract data for Relay
            relay_data = data.val().get("Relay", {})
            relay_state = relay_data.get("State", False)
            
            return temperature, humidity, light_intensity, relay_state
        else:
            return 0.0, 0.0, 0.0, False
    except Exception as e:
        st.error(f"Error fetching data from Firebase: {e}")
        return 0.0, 0.0, 0.0, False


# Main app
st.subheader("Real-Time Sensor Monitoring")

# Fetch data from Firebase
temperature, humidity, light_intensity, relay_state = fetch_firebase_data()

# Display sensor data
st.markdown(f"""
- **Temperature:** {temperature:.2f} °C  
- **Humidity:** {humidity:.2f} %  
- **Light Intensity:** {light_intensity} Lux  
""")

# Predict cocoon yield
def predict_cocoon_yield(temp, hum, light, feed_quality):
    input_data = np.array([[temp, hum, light, feed_quality]])
    return model.predict(input_data)[0]

# Health condition logic
def health_condition(predicted_yield):
    if predicted_yield < 50:
        return "Unhealthy", "#FF4500"  # Red
    elif predicted_yield < 150:
        return "Moderate", "#FFD700"  # Yellow
    else:
        return "Healthy", "#32CD32"  # Green

# Prediction button
if st.button("Predict Cocoon Yield", key="predict_button"):
    predicted_yield = predict_cocoon_yield(temperature, humidity, light_intensity, feeding_quality)
    condition, color = health_condition(predicted_yield)
    
    st.markdown(f"""
    <div style="text-align:center; padding:20px; border-radius:10px; background-color:{color}; color:white; font-weight:bold;">
        Predicted Cocoon Yield: {predicted_yield:.2f} kg  <br>
        Health Condition: {condition}
    </div>
    """, unsafe_allow_html=True)

    # Display Relay State
    relay_state_text = "ON" if relay_state else "OFF"
    relay_state_color = "#32CD32" if relay_state else "#FF4500"
    st.markdown(f"""
    <div style="text-align:center; padding:15px; margin-top:10px; border-radius:10px; background-color:{relay_state_color}; color:white; font-weight:bold;">
        Relay State: {relay_state_text}
    </div>
    """, unsafe_allow_html=True)
