from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union
import pandas as pd
import numpy as np
from house_price_prediction.inference.predictor import HousePricePredictor

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI(title="House Price Prediction API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (Outputs)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
app.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")

# Mount Frontend
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), '../frontend')
app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

# Initialize predictor
predictor = HousePricePredictor()

class HouseFeatures(BaseModel):
    # Simplified Input
    area: Optional[float] = None # GrLivArea
    location: Optional[str] = None # Neighborhood
    bedrooms: Optional[int] = None # BedroomAbvGr
    bathrooms: Optional[int] = None # FullBath
    amenities: Optional[List[str]] = None # List of amenities
    
    # Allow other fields for full control
    class Config:
        extra = "allow"

class PredictionResponse(BaseModel):
    predicted_price: float
    confidence_range: str

@app.post("/predict-price", response_model=PredictionResponse)
def predict_price(features: HouseFeatures):
    data_dict = features.dict(exclude_unset=True)
    
    # Map simplified features to model features
    model_input = {}
    
    # Default values (could be improved with dataset statistics)
    defaults = {
        'OverallQual': 5,
        'YearBuilt': 1980,
        'YrSold': 2010,
        'TotalBsmtSF': 0,
        'GarageCars': 0,
        'GarageArea': 0
    }
    model_input.update(defaults)
    
    # Mapping
    if 'area' in data_dict:
        model_input['GrLivArea'] = data_dict['area']
        model_input['TotalBsmtSF'] = data_dict['area'] * 0.5 # Rough assumption
    
    if 'location' in data_dict:
        # Map "Urban" to a generic neighborhood or zoning if needed
        # For now, pass it as Neighborhood if it matches, else ignore
        model_input['Neighborhood'] = data_dict['location']
    
    if 'bedrooms' in data_dict:
        model_input['BedroomAbvGr'] = data_dict['bedrooms']
    
    if 'bathrooms' in data_dict:
        model_input['FullBath'] = int(data_dict['bathrooms'])
        model_input['HalfBath'] = 1 if data_dict['bathrooms'] % 1 != 0 else 0
    
    if 'amenities' in data_dict and data_dict['amenities']:
        amenities = data_dict['amenities']
        if 'parking' in amenities:
            model_input['GarageCars'] = 2
            model_input['GarageArea'] = 500
        if 'pool' in amenities:
            model_input['PoolArea'] = 100
        if 'fireplace' in amenities:
            model_input['Fireplaces'] = 1
            
    # Merge extra fields (allows overriding)
    for k, v in data_dict.items():
        if k not in ['area', 'location', 'bedrooms', 'bathrooms', 'amenities']:
            model_input[k] = v
            
    try:
        price = predictor.predict(model_input)
        
        # Confidence range (heuristic)
        lower = price * 0.95
        upper = price * 1.05
        confidence = f"+/- 5% (${lower:,.0f} - ${upper:,.0f})"
        
        return {
            "predicted_price": round(price, 2),
            "confidence_range": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to House Price Prediction API"}
