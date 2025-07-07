import streamlit as st
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
import base64
import matplotlib.pyplot as plt

# ------------------ CONFIG ------------------
st.set_page_config(page_title="CNC Tool Wear Prediction", layout="wide", page_icon="🛠️")

# ------------------ CSS ------------------
st.markdown("""
    <style>
        h1, h2, h3, h4, h5, p, .stMarkdown {
            color: #ffffff !important;
            font-weight: bold !important;
            font-size: 18px !important;
        }
        .stAlert-success {
            background-color: rgba(0, 255, 0, 0.15) !important;
            border-left: 0.5rem solid #00ffcc !important;
            color: #00ffcc !important;
            font-weight: bold;
            font-size: 16px;
        }
        .stFileUploader label {
            color: white !important;
            font-size: 16px !important;
            font-weight: bold;
        }
        div[data-testid="stTabs"] > div {
            background-color: rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 6px;
        }
        button[role="tab"] {
            background-color: #222831;
            color: #ffffff !important;
            font-weight: bold;
            border: 2px solid #00adb5;
            border-radius: 8px;
            margin-right: 6px;
            padding: 10px;
            transition: all 0.3s ease;
        }
        button[role="tab"]:hover {
            background-color: #00adb5;
            color: #000000 !important;
        }
        button[aria-selected="true"] {
            background-color: #00adb5;
            color: black !important;
        }
        .dataframe {
            background-color: rgba(0, 0, 0, 0.6);
            color: white !important;
            font-size: 14px;
        }
        section[data-testid="stSidebar"] * {
            color: #000000 !important;
            font-weight: bold !important;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------ BACKGROUND ------------------
def add_bg_from_local(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("/home/ubuntu/Tolwear_prediction/testalize-me-UvZBczaG6rc-unsplash.jpg")


# ------------------ LOAD MODEL & SCALER ------------------
model = load_model("my_lstm_model.h5")
scaler = joblib.load("scaler.pkl")

FEATURES = ['M1_CURRENT_FEEDRATE', 'X1_ActualAcceleration', 'X1_ActualPosition', 'X1_ActualVelocity',
 'X1_CommandAcceleration', 'X1_CommandPosition', 'X1_CommandVelocity', 'X1_CurrentFeedback',
 'X1_DCBusVoltage', 'X1_OutputCurrent', 'X1_OutputVoltage', 'Y1_ActualPosition',
 'Y1_CommandPosition', 'Y1_OutputCurrent', 'clamp_pressure', 'feedrate']

# ------------------ SIDEBAR NAVIGATION ------------------
st.sidebar.title("🧽 Navigation")
selected_tab = st.sidebar.radio("Go to", ["🏠 Home", "🔍 Predict Tool Wear"])

if selected_tab == "🔍 Predict Tool Wear":
    st.sidebar.markdown("### 🛠️ Input CNC Features")
    user_inputs = {}
    for i, feature in enumerate(FEATURES):
        user_inputs[feature] = st.sidebar.number_input(f"{feature}", value=0.0, format="%.4f", key=f"input_{i}")
    predict_button = st.sidebar.button("🔍 Predict", key="predict_button")

if selected_tab == "🏠 Home":
    st.markdown("<h1 style='color: black;'>🛠️ CNC Tool Wear Prediction Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("""
        <div style='background-color: rgba(0,0,0,0.6); padding: 20px; border-radius: 10px;'>
        <h2 style='color: #00ffcc;'>📌 Project Title</h2>
        <p style='font-size: 18px; color: white;'><b>CNC Milling Performance Analysis and Fault Detection</b></p>
        <h3 style='color: #00ffcc;'>🎯 Project Overview</h3>
        <p style='font-size: 16px; color: white;'>
            This project focuses on leveraging deep learning, particularly LSTM models, to monitor and predict tool wear conditions during CNC milling operations.
        </p>
        <h3 style='color: #00ffcc;'>🧠 Skills Gained</h3>
        <ul style='color: white; font-size: 16px;'>
            <li>Data Cleaning and Preprocessing</li><li>Time Series Analysis and Feature Engineering</li>
            <li>Model Evaluation and Performance Metrics</li><li>Deploying Deep Learning Models using AWS and Streamlit</li>
        </ul>
        <h3 style='color: #00ffcc;'>🏭 Domain</h3>
        <p style='font-size: 16px; color: white;'>Manufacturing and Industrial Automation</p>
        <h3 style='color: #00ffcc;'>🧪 What This App Predicts</h3>
        <ul style='color: white; font-size: 16px;'>
            <li><b>Tool Condition</b> – Good / Worn / Damaged</li>
            <li><b>Machining Finalized</b> – Yes / No</li>
            <li><b>Visual Inspection</b> – Passed / Failed</li>
        </ul>
        <h4 style='color: #00ffcc;'>👨‍💻 Developed By:</h4>
        <p style='font-size: 16px; color: white;'>Suriya Vignesh</p>
        </div>
    """, unsafe_allow_html=True)


# ------------------ PREDICTION ------------------
if selected_tab == "🔍 Predict Tool Wear":
    st.markdown("<h2>🎛️ CNC Tool Wear Prediction</h2>", unsafe_allow_html=True)

    if predict_button:
        try:
            input_df = pd.DataFrame([user_inputs])
            input_scaled = scaler.transform(input_df)

            # Simulate sequence by repeating single row input
            sequence_length = 10  # Must match training setup
            input_sequence = np.array([input_scaled] * sequence_length)
            input_sequence = input_sequence.reshape(1, sequence_length, len(FEATURES))

            pred_output = model.predict(input_sequence)
            tool_pred = np.argmax(pred_output[0][0])
            machining_pred = int(np.round(pred_output[1][0][0]))

            # Override logic for visual_pred if tool is worn
            if tool_pred == 1:  # 1 means "Worn"
                visual_pred = 0  # Force to "Failed"
            else:
                visual_pred = int(np.round(pred_output[2][0][0]))

            tool_map = {0: "Unworn", 1: "Worn"}
            machining_map = {0: "No", 1: "Yes"}
            visual_map = {0: "Failed", 1: "Passed"}

            # Get color based on prediction
            tool_color = "#00cc66" if tool_pred == 0 else "#ff4d4d"  # Green if Unworn, Red otherwise
            visual_color = "#00cc66" if visual_pred == 1 else "#ff4d4d"  # Green if Passed, Red if Failed

            # Display predictions with color
            st.markdown("---")
            st.subheader("🔮 Predictions")
            st.markdown(f"🔧 <b>Tool Condition:</b> <span style='color:{tool_color}'>{tool_map[tool_pred]}</span>", unsafe_allow_html=True)
            st.markdown(f"✅ <b>Machining Finalized:</b> <span style='color:#00ffcc'>{machining_map[machining_pred]}</span>", unsafe_allow_html=True)
            st.markdown(f"🧪 <b>Visual Inspection:</b> <span style='color:{visual_color}'>{visual_map[visual_pred]}</span>", unsafe_allow_html=True)



        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")
