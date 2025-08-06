from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import pandas as pd
import logging
from typing import Optional
from app.predict import make_prediction
from app.preprocessing import preprocess_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Car Price Prediction API",
    description="A machine learning API for predicting car prices based on vehicle features",
    version="1.0.0"
)

class CarFeatures(BaseModel):
    """Input schema for car price prediction"""
    symboling: int = Field(..., ge=-3, le=3, description="Insurance risk rating")
    CarName: str = Field(..., description="Car name")
    fueltype: str = Field(..., pattern="^(gas|diesel)$", description="Fuel type")
    aspiration: str = Field(..., pattern="^(std|turbo)$", description="Engine aspiration")
    doornumber: str = Field(..., pattern="^(two|four)$", description="Number of doors")
    carbody: str = Field(..., pattern="^(convertible|hatchback|sedan|wagon|hardtop)$", description="Car body type")
    drivewheel: str = Field(..., pattern="^(rwd|fwd|4wd)$", description="Drive wheel type")
    enginelocation: str = Field(..., pattern="^(front|rear)$", description="Engine location")
    wheelbase: float = Field(..., gt=0, description="Wheelbase in inches")
    carlength: float = Field(..., gt=0, description="Car length in inches")
    carwidth: float = Field(..., gt=0, description="Car width in inches")
    carheight: float = Field(..., gt=0, description="Car height in inches")
    curbweight: int = Field(..., gt=0, description="Curb weight in pounds")
    enginetype: str = Field(..., pattern="^(dohc|dohcv|l|ohc|ohcf|ohcv|rotor)$", description="Engine type")
    cylindernumber: str = Field(..., pattern="^(two|three|four|five|six|eight|twelve)$", description="Number of cylinders")
    enginesize: int = Field(..., gt=0, description="Engine size in cubic inches")
    fuelsystem: str = Field(..., pattern="^(1bbl|2bbl|4bbl|idi|mfi|mpfi|spdi|spfi)$", description="Fuel system type")
    boreratio: float = Field(..., gt=0, description="Bore ratio")
    stroke: float = Field(..., gt=0, description="Stroke")
    compressionratio: float = Field(..., gt=0, description="Compression ratio")
    horsepower: int = Field(..., gt=0, description="Horsepower")
    peakrpm: int = Field(..., gt=0, description="Peak RPM")
    citympg: int = Field(..., gt=0, description="City miles per gallon")
    highwaympg: int = Field(..., gt=0, description="Highway miles per gallon")



class PredictionResponse(BaseModel):
    """Response schema for car price prediction"""
    predicted_price: float = Field(..., description="Predicted car price in USD")
    status: str = Field(default="success", description="Prediction status")
    message: Optional[str] = Field(None, description="Additional information")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Car Price Prediction API",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    try:
        # Test if model can be loaded (basic health check)
        test_data = pd.DataFrame([{
            'symboling': 0, 'CarName': 'toyota corolla', 'fueltype': 'gas', 'aspiration': 'std',
            'doornumber': 'four', 'carbody': 'sedan', 'drivewheel': 'fwd',
            'enginelocation': 'front', 'wheelbase': 100.0, 'carlength': 170.0,
            'carwidth': 65.0, 'carheight': 55.0, 'curbweight': 2500,
            'enginetype': 'ohc', 'cylindernumber': 'four', 'enginesize': 120,
            'fuelsystem': 'mpfi', 'boreratio': 3.0, 'stroke': 3.0,
            'compressionratio': 9.0, 'horsepower': 100, 'peakrpm': 5000,
            'citympg': 25, 'highwaympg': 30
        }])
        
        processed_data = preprocess_data(test_data)
        # This will raise an exception if model loading fails
        _ = make_prediction(processed_data)
        
        return {
            "status": "healthy",
            "model_status": "loaded",
            "timestamp": pd.Timestamp.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy: model loading failed")

@app.post("/predict/", response_model=PredictionResponse)
async def predict_car_price(features: CarFeatures):
    """Predict car price based on input features"""
    try:
        logger.info(f"Received prediction request for car: {features.CarName}")
        
        # Convert Pydantic model to DataFrame
        input_data = pd.DataFrame([features.dict()])
        
        # Preprocess the data (extract brand, clean data)
        processed_data = preprocess_data(input_data)
        
        # Make prediction
        prediction = make_prediction(processed_data)
        
        if prediction is None or len(prediction) == 0:
            raise HTTPException(status_code=500, detail="Prediction failed: no result returned")
        
        predicted_price = float(prediction[0])
        
        logger.info(f"Prediction successful: ${predicted_price:,.2f}")
        
        return PredictionResponse(
            predicted_price=predicted_price,
            status="success",
            message=f"Price prediction completed for {features.CarName}"
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid input data: {str(e)}")
    
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {str(e)}")
        raise HTTPException(status_code=503, detail="Model not available. Please ensure model files are present.")
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    try:
        from app.config import MODEL_PATH, COLUMN_LIST_PATH
        import os
        
        model_exists = os.path.exists(MODEL_PATH)
        columns_exists = os.path.exists(COLUMN_LIST_PATH)
        
        info = {
            "model_path": MODEL_PATH,
            "model_exists": model_exists,
            "columns_path": COLUMN_LIST_PATH,
            "columns_exists": columns_exists,
            "status": "ready" if (model_exists and columns_exists) else "not_ready"
        }
        
        if columns_exists:
            with open(COLUMN_LIST_PATH, 'r') as f:
                columns = [line.strip() for line in f.readlines()]
                info["feature_count"] = len(columns)
                info["features"] = columns  # Show all features
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving model information: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)