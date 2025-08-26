import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import base64
from pathlib import Path

from lc_fetcher import fetch_lightcurve
from process_and_reshape import process_lightcurve
from star_info import get_star_info

MODEL_PATH = "models/model.pth"
TEMP_PATH = "models/temperature.npy"
LOGO_PATH = "logo.png"
BACKGROUND_PATH = "background.jpeg"

def get_image_as_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

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

st.set_page_config(
    page_title="Luminara",
    layout="wide",
    page_icon="✨"
)

def add_bg_and_css(background_image_b64):
    """Injects CSS for static background, custom fonts, and modern UI."""
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Audiowide&family=Roboto:wght@400;700&family=Orbitron:wght@700&display=swap');

    /* --- HIDE THE STREAMLIT HEADER --- */
    header {{
        visibility: hidden;
    }}

    /* --- STATIC IMAGE BACKGROUND (USING BASE64)--- */
    .stApp {{
        background-image: url("data:image/jpeg;base64,{background_image_b64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    
    .stApp::before {{
        content: "";
        position: fixed;
        left: 0; right: 0; top: 0; bottom: 0;
        background-color: rgba(0, 0, 0, 0.5);
        z-index: -1;
    }}

    /* --- Base Font --- */
    html, body, [class*="st-"] {{
        font-family: 'Roboto', sans-serif;
        font-size: 16px;
    }}
    
    /* --- Header font --- */
    h3 {{
        font-family: 'Orbitron', sans-serif !important;
        color: #E0E0E0;
        border-bottom: 2px solid #9370DB;
        padding-bottom: 10px;
        margin-top: 10px;
        text-transform: uppercase;
        letter-spacing: 2px;
    }}

    /* --- Modern Obsidian Card Effect --- */
    .glass-card {{
        background: rgba(10, 10, 20, 0.8);
        backdrop-filter: blur(5px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 25px;
        margin-bottom: 25px;
    }}
    
    /* --- Custom Metrics Styling --- */
    .info-container {{
        display: flex; flex-wrap: wrap; gap: 15px; margin-bottom: 15px;
    }}
    .info-metric {{
        flex: 1 1 150px; text-align: center; padding: 12px; border-radius: 10px;
        background: rgba(147, 112, 219, 0.15);
    }}
    .info-metric-label {{ font-size: 1rem; color: #B0B0B0; margin-bottom: 5px; }}
    .info-metric-value {{ font-size: 1.3rem; font-weight: bold; color: #FFFFFF; word-wrap: break-word; }}

    /* --- MODIFIED: Styles for Circular Progress Indicator --- */
    .progress-circle {{
        position: relative;
        width: 220px;
        height: 220px;
        border-radius: 50%;
        display: grid;
        place-items: center;
        margin: 0 auto 15px auto; /* Center the circle */
        box-shadow: 0 0 25px rgba(147, 112, 219, 0.4);
    }}
    .progress-circle::before {{
        content: "";
        position: absolute;
        height: 86%;
        width: 86%;
        background-color: rgb(10, 10, 20); /* Match the glass-card bg */
        border-radius: 50%;
    }}
    .progress-value {{
        position: relative;
        font-family: 'Audiowide', cursive;
        font-size: 4rem; /* Enlarge the score text */
    }}
    .confidence-verdict {{
        font-size: 1.2rem;
        text-align: center;
    }}
    </style>
    """, unsafe_allow_html=True)

astro_template = {
    "layout": go.Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(10, 25, 47, 0.85)',
        font={'color': '#f0f0f0', 'family': 'Roboto', 'size': 14},
        xaxis={'gridcolor': 'rgba(255, 255, 255, 0.1)', 'linecolor': 'rgba(255, 255, 255, 0.3)'},
        yaxis={'gridcolor': 'rgba(255, 255, 255, 0.1)', 'linecolor': 'rgba(255, 255, 255, 0.3)'},
        legend={'bgcolor': 'rgba(0,0,0,0)', 'font': {'color': '#f0f0f0'}},
    )
}

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN()
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    temperature_np = np.load(TEMP_PATH)
    temperature = torch.tensor(temperature_np, dtype=torch.float32).to(device)
    return model, temperature, device

def predict_with_saliency(model, flux, temperature, device):
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

try:
    logo_b64 = get_image_as_base64(LOGO_PATH)
    bg_b64 = get_image_as_base64(BACKGROUND_PATH)
except FileNotFoundError:
    st.error("Asset files not found. Please ensure 'logo.png' and 'background.jpeg' are in the same folder as your script.")
    st.stop()
    
add_bg_and_css(bg_b64)
model, temperature, device = load_model()

st.markdown(
    f"""
    <div style="text-align: center; padding-top: 20px;">
        <img src="data:image/png;base64,{logo_b64}" alt="Luminara Logo" width="600">
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<div style='text-align: center; color: #B0B0B0; font-size: 1.2rem; margin-bottom: 40px;'>Unveiling hidden worlds through the whispers of starlight.</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    with st.form(key='tic_form'):
        tic_id = st.text_input("Enter TIC ID", placeholder="e.g., TIC 92226327", label_visibility="collapsed")
        analyze_button = st.form_submit_button("Analyze Candidate", use_container_width=True)

if analyze_button and tic_id:
    with st.spinner(f"Contacting Deep Space Network... Analyzing {tic_id}..."):
        try:
            raw_csv = fetch_lightcurve(tic_id)
            processed_csv = process_lightcurve(raw_csv)
            df = pd.read_csv(processed_csv)
            flux = df.drop("LABEL", axis=1).values.flatten()
            
            confidence, saliency = predict_with_saliency(model, flux, temperature, device)
            star_info = get_star_info(tic_id)
            
            st.success(f"Analysis complete for {tic_id}!")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown('<div class="glass-card confidence-card">', unsafe_allow_html=True)
                
                if confidence > 0.8: color, verdict = "#2EFEF7", "High Confidence"
                elif confidence > 0.5: color, verdict = "#FFD700", "Potential Candidate"
                else: color, verdict = "#FF6347", "Unlikely Candidate"
                
                progress_angle = confidence * 360
                
                progress_html = f"""
                <div class="progress-circle" style="background: conic-gradient({color} {progress_angle}deg, rgba(147, 112, 219, 0.15) 0deg);">
                    <div class="progress-value" style="color:{color}; text-shadow: 0 0 15px {color}80;">{confidence:.0%}</div>
                </div>
                <p class="confidence-verdict">{verdict}</p>
                """
                st.markdown(progress_html, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown('<h3>Stellar Information</h3>', unsafe_allow_html=True)
                info_html = f"""<div class="info-container">
                    <div class="info-metric"><div class="info-metric-label">Star Name</div><div class="info-metric-value">{star_info.get('starName', 'N/A')}</div></div>
                    <div class="info-metric"><div class="info-metric-label">TIC ID</div><div class="info-metric-value">{tic_id}</div></div>
                    <div class="info-metric"><div class="info-metric-label">Right Ascension</div><div class="info-metric-value">{star_info.get('ra', 'N/A')}</div></div>
                    <div class="info-metric"><div class="info-metric-label">Declination</div><div class="info-metric-value">{star_info.get('dec', 'N/A')}</div></div>
                    <div class="info-metric"><div class="info-metric-label">Spectral Type</div><div class="info-metric-value">{star_info.get('spectralType', 'N/A')}</div></div>
                </div>"""
                st.markdown(info_html, unsafe_allow_html=True)
                
                st.markdown("<hr style='border-color: rgba(147, 112, 219, 0.3);'>", unsafe_allow_html=True)
                st.markdown("""
                **Glossary:**
                - **RA & Dec:** Celestial coordinates, like longitude and latitude, that locate the star in the sky.
                - **Spectral Type:** Classifies a star by its temperature (O, B, A, F, G, K, M) from hottest to coolest.
                """)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("<h3>Light Curve Analysis</h3>", unsafe_allow_html=True)
            st.markdown("This plot shows the star's brightness over time. A brief, periodic dip (a **'transit'**) can indicate a planet passing in front of its star.")
            fig_lc = go.Figure(data=go.Scatter(y=flux, mode='lines', line=dict(color='#2EFEF7', width=2)))
            fig_lc.update_layout(template=astro_template, title="Light Curve (Flux vs. Time)", xaxis_title='Time Index', yaxis_title='Normalized Flux', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig_lc, use_container_width=True)

            st.markdown("<h3>CNN Saliency Map</h3>", unsafe_allow_html=True)
            st.markdown("This chart reveals **what the AI model focused on**. The highlighted rectangular area shows the data points most influential in its decision.")
            
            fig_sal = go.Figure(data=go.Scatter(y=flux, mode='lines', line=dict(color='#2EFEF7')))
            max_saliency_idx = np.argmax(saliency)
            fig_sal.add_shape(type="rect", xref="x", yref="paper", x0=max_saliency_idx - 20, x1=max_saliency_idx + 20, y0=0, y1=1, fillcolor="rgba(255, 215, 0, 0.4)", line=dict(width=0))
            fig_sal.update_layout(template=astro_template, title="Model Attention Highlighted", xaxis_title='Time Index', yaxis_title='Flux', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig_sal, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("<h3>Final Summary & Conclusion</h3>", unsafe_allow_html=True)
            if confidence > 0.8: summary_text = f"**Verdict: Strong Candidate.** 🟢\n\nWith a high confidence of **{confidence:.1%}**, the AI found strong evidence of a transit. The saliency map confirms the model focused on a distinct dip in the light curve. This is a prime candidate for follow-up observation."
            elif confidence > 0.5: summary_text = f"**Verdict: Potential Candidate.** 🟡\n\nThe model returned a score of **{confidence:.1%}**, indicating some transit-like features but with potential ambiguity. This could be due to noise or a weak signal. This candidate requires further investigation."
            else: summary_text = f"**Verdict: Unlikely Candidate.** 🔴\n\nA low score of **{confidence:.1%}** suggests the model found no clear evidence of a transit. The light curve may be dominated by stellar variability or noise. It's unlikely this star hosts a transiting exoplanet based on this data."
            st.markdown(summary_text)
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
elif analyze_button and not tic_id:
    st.error("Please enter a TIC ID to begin analysis.")