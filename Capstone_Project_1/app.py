# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# Load the trained model
MODEL_PATH = "equipment_output_model_best.joblib"   # adjust if in /mnt/data/
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# Define input schema including all features
class InputFeatures(BaseModel):
    Injection_Temperature: float
    Injection_Pressure: float
    Cycle_Time: float
    Cooling_Time: float
    Material_Viscosity: float
    Ambient_Temperature: float
    Machine_Age: float
    Operator_Experience: float
    Maintenance_Hours: float
    Shift: str
    Machine_Type: str
    Material_Grade: str
    Day_of_Week: str
    Efficiency_Score: float
    Machine_Utilization: float

app = FastAPI(title="Manufacturing Equipment Output Prediction API")

@app.get("/")
def home():
    return {"message": "Welcome to the Equipment Output Prediction API"}

@app.post("/predict")
def predict(features: InputFeatures):
    # Convert input to DataFrame
    data = pd.DataFrame([features.dict()])

    # Compute engineered features if required
    if "Temperature_Pressure_Ratio" in model.named_steps["preprocessor"].transformers_[0][2]:
        data["Temperature_Pressure_Ratio"] = data["Injection_Temperature"] / data["Injection_Pressure"]
    if "Total_Cycle_Time" in model.named_steps["preprocessor"].transformers_[0][2]:
        data["Total_Cycle_Time"] = data["Cycle_Time"] + data["Cooling_Time"]

    # Predict
    prediction = model.predict(data)
    return {"Predicted_Parts_Per_Hour": float(prediction[0])}
