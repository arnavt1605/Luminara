from flask import Flask, render_template, request, redirect, url_for, jsonify
import torch
import torch.nn as nn
import joblib
import pandas as pd
import numpy as np

from lc_fetcher import fetch_lightcurve
from process_and_reshape import process_lightcurve
from star_info import get_star_info

class CNN(nn.Module):
    def __init__(self, input_length):
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
    
MODEL_PATH = "models/model.pth"
SCALER_PATH = "models/scaler.pkl"
TEMP_PATH = "models/temperature.npy"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

scaler = joblib.load(SCALER_PATH)
temperature = np.load(TEMP_PATH)

def predict_with_saliency(model, flux, temperature):

    flux_scaled = scaler.transform([flux])[0]
    x = torch.tensor(flux_scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    x.requires_grad = True

    logits = model(x) / temperature
    prob = torch.sigmoid(logits)
    model.zero_grad()
    logits.backward()
    saliency = x.grad.abs().squeeze().detach().cpu().numpy()

    return float(prob.item()), saliency.tolist()

app = Flask(__name__)
@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("homepage.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    tic_id = request.form.get("tic_id")

    if not tic_id:
        return render_template("homepage.html", error="Please enter a TIC ID")

    raw_csv = fetch_lightcurve(tic_id)
    if raw_csv is None:
        return render_template("homepage.html", error=f"Could not fetch data for {tic_id}")

    processed_csv = process_lightcurve(raw_csv)
    if processed_csv is None:
        return render_template("homepage.html", error=f"Processing failed for {tic_id}")

    df = pd.read_csv(processed_csv)
    flux = df.drop("LABEL", axis=1).values.squeeze()

    confidence, saliency = predict_with_saliency(model, flux, temperature)

    star_info = get_star_info(tic_id)

    return render_template(
        "information.html",
        tic_id=tic_id,
        confidence=confidence,
        flux=flux.tolist(),
        saliency=saliency,
        star_info=star_info
    )

if __name__ == "__main__":
    app.run(debug=True)