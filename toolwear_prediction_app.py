import streamlit as st
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
import base64

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

add_bg_from_local("D:/guvi/Final_project/testalize-me-UvZBczaG6rc-unsplash.jpg")

# ------------------ LOAD MODEL & SCALER ------------------
model = load_model("my_lstm_model.h5")
scaler = joblib.load("scaler.pkl")

FEATURES = [
    'feedrate', 'Y1_OutputCurrent', 'clamp_pressure', 'X1_OutputCurrent',
    'M1_CURRENT_FEEDRATE', 'X1_CommandPosition', 'X1_ActualPosition',
    'X1_OutputVoltage', 'Y1_CommandPosition', 'Y1_ActualPosition',
    'X1_ActualVelocity', 'X1_ActualAcceleration', 'X1_CommandVelocity',
    'X1_CommandAcceleration', 'X1_CurrentFeedback', 'X1_DCBusVoltage'
]

# ------------------ TABS ------------------
tabs = st.tabs(["🏠 Home", "🔍 Predict Tool Wear"])

# ------------------ HOME ------------------
with tabs[0]:
    st.markdown("<h1 style='color: white;'>🛠️ CNC Tool Wear Prediction Dashboard</h1>", unsafe_allow_html=True)

    st.markdown("""
            <div style='background-color: rgba(0,0,0,0.6); padding: 20px; border-radius: 10px;'>
            <h2 style='color: #00ffcc;'>📌 Project Title</h2>
            <p style='font-size: 18px; color: white;'><b>CNC Milling Performance Analysis and Fault Detection</b></p>

            <h3 style='color: #00ffcc;'>🎯 Project Overview</h3>
            <p style='font-size: 16px; color: white;'>
                This project focuses on leveraging deep learning, particularly LSTM models, to monitor and predict tool wear conditions during CNC milling operations. 
                The goal is to enhance machine performance, enable predictive maintenance, and detect potential faults before failure. By analyzing real-time CNC data, 
                the model can determine the condition of the tool, whether machining is complete, and if the part passes visual inspection.
            </p>

            <h3 style='color: #00ffcc;'>🧠 Skills Gained</h3>
            <ul style='color: white; font-size: 16px;'>
                <li>Data Cleaning and Preprocessing</li>
                <li>Exploratory Data Analysis (EDA)</li>
                <li>Feature Engineering for Neural Networks</li>
                <li>Building and Training Artificial Neural Networks (ANNs)</li>
                <li>Model Evaluation and Optimization</li>
                <li>MYSQL Usage and Integration</li>
                <li>Deploying Deep Learning Models using AWS and Streamlit</li>
                <li>Time Series Analysis and Feature Engineering</li>
                <li>Predictive Maintenance and Fault Detection</li>
                <li>Model Evaluation and Performance Metrics</li>
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
with tabs[1]:
    st.markdown("<h2>📁 Upload your CNC experiment CSV</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a CSV file with 16 features", type=['csv'])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            if not set(FEATURES).issubset(df.columns):
                st.error("❌ CSV does not contain all required feature columns.")
                st.markdown(f"**Expected Columns:** {FEATURES}")
            else:
                st.success("✅ File uploaded successfully!")
                st.dataframe(df.head())

                scaled = scaler.transform(df[FEATURES])
                reshaped = scaled.reshape(1, scaled.shape[0], scaled.shape[1])
                pred_output = model.predict(reshaped)

                # Predictions from model output
                tool_pred = np.argmax(pred_output[0][0])
                machining_pred = int(np.round(pred_output[1][0][0]))
                visual_pred = int(np.round(pred_output[2][0][0]))

                # Labels
                tool_map = {0: "Good", 1: "Worn", 2: "Damaged"}
                machining_map = {0: "No", 1: "Yes"}
                visual_map = {0: "Failed", 1: "Passed"}

                # Results
                st.markdown("---")
                st.subheader("🔮 Predictions")
                st.markdown(f"🔧 <b>Tool Condition:</b> <span style='color:#00ffcc'>{tool_map[tool_pred]}</span>", unsafe_allow_html=True)
                st.markdown(f"✅ <b>Machining Finalized:</b> <span style='color:#00ffcc'>{machining_map[machining_pred]}</span>", unsafe_allow_html=True)
                st.markdown(f"🧪 <b>Visual Inspection:</b> <span style='color:#00ffcc'>{visual_map[visual_pred]}</span>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"⚠️ Error processing file: {e}")
