from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Rossmann Sales Prediction API")

# 1. Load Model & Schema
print("⏳ Loading Model & Schema...")
model = joblib.load("models/production_model.pkl")
model_columns = joblib.load("models/model_columns.pkl") 
print("✅ Ready to predict!")

class SalesInput(BaseModel):
    Store: int
    DayOfWeek: int
    Promo: int
    SchoolHoliday: int
    StoreType: str
    Assortment: str
    CompetitionDistance: float
    Promo2: int

@app.post("/predict")
def predict_sales(data: SalesInput):
    # 1. Convert Input to DataFrame
    df = pd.DataFrame([data.dict()])
    
    # 2. Force Store to be Open (Crucial Fix)
    df['Open'] = 1
    
    # 3. Preprocessing
    df_processed = pd.get_dummies(df)
    
    # 4. Align Columns
    df_final = df_processed.reindex(columns=model_columns, fill_value=0)
    
    # 5. Predict
    try:
        prediction = model.predict(df_final)
        return {"predicted_sales": float(prediction[0])}
    except Exception as e:
        return {"error": str(e)}