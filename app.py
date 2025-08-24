import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# (Assuming these helper files are in the same directory)
from lc_fetcher import fetch_lightcurve
from process_and_reshape import process_lightcurve
from star_info import get_star_info


# --- CONFIGURATION ---
MODEL_PATH = "models/model.pth"
TEMP_PATH = "models/temperature.npy"

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack= nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=11, padding=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier=  nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        x=self.stack(x)
        x= self.classifier(x).squeeze(1)
        return x

# --- 2. Define Helper Functions ---
@st.cache_resource
def load_model():
    """Loads the trained model and temperature from files."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN()
    checkpoint = torch.load("models/model.pth", map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    temperature_np = np.load(TEMP_PATH)
    temperature = torch.tensor(temperature_np, dtype=torch.float32).to(device)
    return model, temperature, device

def predict_with_saliency(model, flux, temperature, device):
    """Performs prediction with on-the-fly normalization."""
    flux_np = np.array(flux, dtype=np.float32)
    median = np.median(flux_np)
    flux_centered = flux_np - median
    mad = np.median(np.abs(flux_centered))
    if mad == 0: mad = 1.0
    flux_scaled = flux_centered / mad
    
    x = torch.tensor(flux_scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    x.requires_grad = True
    logits = model(x)
    prob = torch.sigmoid(logits / temperature)
    model.zero_grad()
    logits.backward()
    confidence = prob.detach().cpu().item()
    saliency = x.grad.detach().abs().squeeze().cpu().numpy()
    return confidence, saliency.tolist()

# --- 3. Build the User Interface ---

# --- Page Configuration ---
st.set_page_config(page_title="Exo-Planet Analyzer", layout="wide", page_icon="🛰️")

# --- Load Model ---
model, temperature, device = load_model()

# --- Header and Title ---
st.title("🛰️ Exo-Planet Saliency Analyzer")
st.write("Enter a TESS Input Catalog (TIC) ID to analyze its light curve for exoplanet candidates.")

# --- Input Form ---
tic_id = st.text_input("Enter TIC ID (e.g., 'TIC 92226327')", "")

if st.button("Analyze"):
    if not tic_id:
        st.error("Please enter a TIC ID.")
    else:
        with st.spinner(f"Fetching and analyzing data for {tic_id}..."):
            try:
                # --- Data Fetching and Prediction ---
                raw_csv = fetch_lightcurve(tic_id)
                processed_csv = process_lightcurve(raw_csv)
                df = pd.read_csv(processed_csv)
                flux = df.drop("LABEL", axis=1).values.flatten()
                
                confidence, saliency = predict_with_saliency(model, flux, temperature, device)
                star_info = get_star_info(tic_id)
                
                st.success(f"Analysis complete for {tic_id}!")

                # --- Info Bar ---
                cols = st.columns(5)
                cols[0].metric("Star Name", star_info.get('starName', 'N/A'))
                cols[1].metric("TIC ID", tic_id)
                cols[2].metric("RA", star_info.get('ra', 'N/A'))
                cols[3].metric("Dec", star_info.get('dec', 'N/A'))
                cols[4].metric("Spectral Type", star_info.get('spectralType', 'N/A'))
                
                # --- Plotly Chart Layout ---
                plot_layout = {
                    'paper_bgcolor': 'rgba(0,0,0,0)',
                    'plot_bgcolor': 'rgba(42, 42, 60, 0.8)',
                    'font': {'color': '#f0f0f0', 'family': 'sans-serif'},
                    'xaxis': {'gridcolor': 'rgba(255, 255, 255, 0.1)'},
                    'yaxis': {'gridcolor': 'rgba(255, 255, 255, 0.1)'}
                }

                # --- Light Curve Plot ---
                st.subheader("Light Curve (Flux vs. Time)")
                fig_lc = go.Figure(data=go.Scatter(y=flux, mode='lines', line=dict(color='#f5f5dc')))
                # Add the rangeslider here
                fig_lc.update_layout(**plot_layout, xaxis_title='Time Index', yaxis_title='Flux', xaxis_rangeslider_visible=True)
                st.plotly_chart(fig_lc, use_container_width=True)
                
                # --- Saliency Overlay Plot ---
                st.subheader("CNN Saliency Overlay")
                fig_sal = go.Figure(data=go.Scatter(y=flux, mode='lines', line=dict(color='#f5f5dc')))
                max_saliency_idx = np.argmax(saliency)
                fig_sal.add_shape(type="rect",
                                xref="x", yref="paper",
                                x0=max_saliency_idx - 10, x1=max_saliency_idx + 10, y0=0, y1=1,
                                fillcolor="rgba(255, 204, 0, 0.4)", line=dict(width=0))
                # Add the rangeslider here as well
                fig_sal.update_layout(**plot_layout, title="Model Attention Highlighted", xaxis_title='Time Index', yaxis_title='Flux', xaxis_rangeslider_visible=True)
                st.plotly_chart(fig_sal, use_container_width=True)

                # --- Bottom Row: Confidence Score and Histogram ---
                col1, col2 = st.columns([1, 2]) # Give more space to the histogram

                with col1:
                    st.subheader("Exoplanet Confidence Score")
                    # Using a Plotly Gauge for the score circle
                    gauge_color = "#4CAF50" if confidence > 0.7 else "#FFC107" if confidence > 0.4 else "#F44336"
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = confidence * 100,
                        number = {'suffix': "%", 'font': {'size': 40}},
                        gauge = {'axis': {'range': [None, 100]},
                                 'bar': {'color': gauge_color},
                                 'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50}},
                        domain = {'x': [0, 1], 'y': [0, 1]}
                    ))
                    fig_gauge.update_layout(paper_bgcolor = "rgba(0,0,0,0)", font = {'color': "white"}, height=250)
                    st.plotly_chart(fig_gauge, use_container_width=True)

                with col2:
                    st.subheader("Brightness Distribution")
                    fig_hist = go.Figure(data=[go.Histogram(x=flux, marker_color='#f5f5dc')])
                    fig_hist.update_layout(**plot_layout, xaxis_title='Flux', yaxis_title='Count')
                    st.plotly_chart(fig_hist, use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.exception(e) # This will print the full traceback