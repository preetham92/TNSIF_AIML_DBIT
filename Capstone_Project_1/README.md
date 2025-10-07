# Data Science Capstone Project: Manufacturing Equipment Output Prediction

## 📌 Project Overview
This project predicts **hourly machine output (Parts_Per_Hour)** for an injection molding manufacturing process using a **Linear Regression model**.  
By leveraging operating parameters such as temperature, pressure, cycle time, material properties, and machine settings, the model helps the production team:
- Optimize machine parameters
- Improve efficiency
- Plan production schedules
- Detect underperforming machines

---

## 📂 Dataset Information
- **Name**: `manufacturing_dataset_1000_samples.csv`
- **Size**: 1000 records
- **Target Variable**: `Parts_Per_Hour`
- **Features**:
  - Injection_Temperature  
  - Injection_Pressure  
  - Cycle_Time  
  - Cooling_Time  
  - Material_Viscosity  
  - Ambient_Temperature  
  - Machine_Age  
  - Operator_Experience  
  - Maintenance_Hours  
  - Shift (Day/Night/Evening)  
  - Machine_Type (Type_A / Type_B)  
  - Material_Grade (Economy/Standard/Premium)  
  - Day_of_Week (Monday–Sunday)  
  - Efficiency_Score  
  - Machine_Utilization  

**Derived Features (computed in API):**
- `Temperature_Pressure_Ratio = Injection_Temperature / Injection_Pressure`
- `Total_Cycle_Time = Cycle_Time + Cooling_Time`

---

## ⚙️ Tools & Technologies
- Python (3.11)
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn
- FastAPI (for API deployment)
- Uvicorn (server)
- Docker (for containerization)

---

## 🛠️ Project Workflow
1. **EDA & Preprocessing**
   - Handle missing values
   - Encode categorical variables
   - Scale numerical features
   - Feature engineering
2. **Model Development**
   - Baseline Linear Regression
   - Train-test split (80/20)
   - Evaluate with RMSE, MSE, R²
3. **Deployment**
   - Save trained model with `joblib`
   - Build FastAPI app (`app.py`)
   - Run using Uvicorn
   - Containerize with Docker (optional)

---

## 🚀 How to Run the Project

### 🔹 Option 1: Run Locally (Recommended for Development)

#### 1. Clone Repository
```bash
git clone <your-repo-link>
cd Capstone_Project_1


uvicorn app:app --reload --host 0.0.0.0 --port 8000

API Root: http://0.0.0.0:8000

Swagger Docs: http://0.0.0.0:8000/docs

Redoc Docs: http://0.0.0.0:8000/redoc