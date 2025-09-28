import pickle
import joblib
import numpy as np
from typing import Dict, Any, Tuple
import xgboost as xgb
from app.config import settings
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class RiskPredictionModel:
    def __init__(self):
        self.model = None
        self.model_metadata = {}
        self.feature_names = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']
        self.model_loaded = False
    
    def load_model(self, model_path: str) -> bool:
        """Load XGBoost model from pickle or joblib file"""
        try:
            model_file = Path(model_path)
            if not model_file.exists():
                logger.error(f"Model file not found: {model_path}")
                return False
            
            if model_path.endswith('.pkl'):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
            elif model_path.endswith('.joblib'):
                self.model = joblib.load(model_path)
            else:
                logger.error("Unsupported model format. Use .pkl or .joblib")
                return False
            
            # Extract model metadata
            self.model_metadata = {
                'model_type': type(self.model).__name__,
                'features': self.feature_names,
                'model_path': model_path,
                'loaded_at': np.datetime64('now')
            }
            
            self.model_loaded = True
            logger.info(f"Model loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {str(e)}")
            self.model_loaded = False
            return False
    
    def predict(self, features: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
        """Predict risk level and return label, probability, and feature importances"""
        if not self.model_loaded or not self.model:
            raise Exception("Model not loaded. Call load_model() first.")
        
        try:
            # Prepare features in correct order
            feature_array = np.array([[features['Age'], features['SystolicBP'], 
                                     features['DiastolicBP'], features['BS'], 
                                     features['BodyTemp'], features['HeartRate']]])
            
            # Predict probability
            if hasattr(self.model, 'predict_proba'):
                probability = self.model.predict_proba(feature_array)[0, 1]
            else:
                # For models that don't have predict_proba
                raw_pred = self.model.predict(feature_array)[0]
                probability = float(raw_pred)
            
            # Determine risk label with threshold
            risk_threshold = 0.5
            risk_label = "high risk" if probability >= risk_threshold else "low risk"
            
            # Calculate feature importances
            feature_importances = self._calculate_feature_importance(features, probability)
            
            logger.debug(f"Prediction completed - Risk: {risk_label}, Probability: {probability:.3f}")
            return risk_label, probability, feature_importances
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise Exception(f"Prediction failed: {str(e)}")
    
    def _calculate_feature_importance(self, features: Dict[str, float], probability: float) -> Dict[str, float]:
        """Calculate feature importance scores"""
        try:
            # Use model's built-in feature importance if available
            if hasattr(self.model, 'feature_importances_'):
                importances = dict(zip(self.feature_names, self.model.feature_importances_))
            else:
                # Fallback: simplified importance based on deviation from normal ranges
                normal_ranges = {
                    'Age': 30, 'SystolicBP': 120, 'DiastolicBP': 80, 
                    'BS': 5.5, 'BodyTemp': 37.0, 'HeartRate': 70
                }
                
                importances = {}
                for feature, value in features.items():
                    normal_value = normal_ranges.get(feature)
                    if normal_value:
                        deviation = abs(value - normal_value) / normal_value
                        importances[feature] = min(deviation, 1.0)  # Cap at 1.0
                
                # Normalize to sum to 1
                total = sum(importances.values())
                if total > 0:
                    importances = {k: v/total for k, v in importances.items()}
            
            return importances
            
        except Exception as e:
            logger.warning(f"Feature importance calculation failed: {e}")
            return {feature: 0.0 for feature in self.feature_names}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata and information"""
        return {
            'loaded': self.model_loaded,
            'metadata': self.model_metadata,
            'feature_names': self.feature_names,
            'risk_threshold': 0.5
        }
    
    def validate_features(self, features: Dict[str, float]) -> bool:
        """Validate input features before prediction"""
        try:
            # Check all required features are present
            for feature in self.feature_names:
                if feature not in features:
                    logger.error(f"Missing feature: {feature}")
                    return False
            
            # Validate feature ranges
            if not (15 <= features['Age'] <= 50):
                logger.error(f"Age out of range: {features['Age']}")
                return False
            
            if not (70 <= features['SystolicBP'] <= 200):
                logger.error(f"SystolicBP out of range: {features['SystolicBP']}")
                return False
            
            if not (40 <= features['DiastolicBP'] <= 130):
                logger.error(f"DiastolicBP out of range: {features['DiastolicBP']}")
                return False
            
            if not (3.0 <= features['BS'] <= 30.0):
                logger.error(f"BS out of range: {features['BS']}")
                return False
            
            if not (35.0 <= features['BodyTemp'] <= 42.0):
                logger.error(f"BodyTemp out of range: {features['BodyTemp']}")
                return False
            
            if not (40 <= features['HeartRate'] <= 150):
                logger.error(f"HeartRate out of range: {features['HeartRate']}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Feature validation error: {e}")
            return False

# Global model instance
risk_model = RiskPredictionModel()

def initialize_model() -> bool:
    """Initialize the ML model on application startup"""
    try:
        success = risk_model.load_model(settings.MODEL_PATH)
        if success:
            logger.info("ML model initialized successfully")
        else:
            logger.error("ML model initialization failed")
        return success
    except Exception as e:
        logger.error(f"ML model initialization error: {e}")
        return False