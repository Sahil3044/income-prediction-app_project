
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model
try:
    model_package = joblib.load("deployment_model.joblib")
    model = model_package['model']
    preprocessor = model_package['preprocessor']
    feature_names = model_package['feature_names']
    model_info = model_package['model_info']
    logger.info(f"Model loaded successfully: {model_info['name']}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

app = FastAPI(
    title="Adult Income Prediction API",
    description="API for predicting whether income exceeds $50K/year",
    version="1.0.0"
)

# Define input data model
class PredictionInput(BaseModel):
    age: int
    workclass: str
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    income_class: str
    confidence: str

class ModelInfo(BaseModel):
    name: str
    f1_score: float
    roc_auc: float
    accuracy: float

@app.get("/")
async def root():
    return {
        "message": "Adult Income Prediction API",
        "status": "active",
        "model": model_info['name']
    }

@app.get("/model-info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the deployed model"""
    return ModelInfo(
        name=model_info['name'],
        f1_score=model_info['f1_score'],
        roc_auc=model_info['roc_auc'],
        accuracy=model_info['accuracy']
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: PredictionInput):
    """Make a prediction"""
    try:
        # Convert input to DataFrame
        input_dict = {
            'age': [input_data.age],
            'workclass': [input_data.workclass],
            'education-num': [input_data.education_num],
            'marital-status': [input_data.marital_status],
            'occupation': [input_data.occupation],
            'relationship': [input_data.relationship],
            'race': [input_data.race],
            'sex': [input_data.sex],
            'capital-gain': [input_data.capital_gain],
            'capital-loss': [input_data.capital_loss],
            'hours-per-week': [input_data.hours_per_week],
            'native-country': [input_data.native_country]
        }

        # Add engineered features
        input_dict['age_group'] = pd.cut([input_data.age],
                                       bins=[0, 25, 35, 45, 55, 65, 100],
                                       labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])[0]

        input_dict['hours_category'] = pd.cut([input_data.hours_per_week],
                                            bins=[0, 35, 40, 50, 100],
                                            labels=['Part-time', 'Full-time', 'Overtime', 'Double-time'])[0]

        input_dict['capital_change'] = [input_data.capital_gain - input_data.capital_loss]
        input_dict['is_us'] = [1 if input_data.native_country == 'United-States' else 0]
        input_dict['has_capital_activity'] = [1 if (input_data.capital_gain > 0 or input_data.capital_loss > 0) else 0]

        # Education level mapping
        education_mapping = {
            'Preschool': 'Elementary', '1st-4th': 'Elementary', '5th-6th': 'Elementary',
            '7th-8th': 'Middle', '9th': 'Middle', '10th': 'Middle',
            '11th': 'High', '12th': 'High', 'HS-grad': 'High',
            'Some-college': 'College', 'Assoc-acdm': 'College', 'Assoc-voc': 'College',
            'Bachelors': 'University', 'Masters': 'Graduate', 'Prof-school': 'Graduate',
            'Doctorate': 'PhD'
        }
        input_dict['education_level'] = [education_mapping.get(input_data.education, 'Unknown')]

        # Create DataFrame
        input_df = pd.DataFrame(input_dict)

        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        # Determine confidence level
        if probability > 0.8 or probability < 0.2:
            confidence = "high"
        elif probability > 0.6 or probability < 0.4:
            confidence = "medium"
        else:
            confidence = "low"

        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            income_class=">50K" if prediction == 1 else "<=50K",
            confidence=confidence
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
