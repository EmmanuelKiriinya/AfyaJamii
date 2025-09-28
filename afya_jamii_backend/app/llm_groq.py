import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class AfyaJamiiLLM:
    def __init__(self):
        self.llm = None
        self.memory = None
        self.chain = None
        self.initialize_llm()
    
    def initialize_llm(self):
        """Initialize Groq LLM with configuration from settings"""
        try:
            if not settings.GROQ_API_KEY or settings.GROQ_API_KEY == "your-groq-api-key-here":
                logger.error("GROQ_API_KEY not configured")
                return
            
            self.llm = ChatGroq(
                model=settings.LLM_MODEL_NAME,
                temperature=settings.LLM_TEMPERATURE,
                api_key=settings.GROQ_API_KEY
            )
            
            # Create prompt template
            template = """
You are Afya Jamii AI, a clinical decision-support and maternal nutrition assistant for Kenyan pregnant and postnatal mothers and general users seeking nutrition advice. Your role is to interpret outputs from an XGBoost risk-prediction model and provide clear, evidence-based, actionable clinical and nutrition guidance tailored to Kenyan context.

⚠️ Important: The underlying ML model has been trained on these patient variables:
[Age, SystolicBP, DiastolicBP, BS (Blood Sugar), BodyTemp, HeartRate]

Interpret the ML model output carefully:
- 0 = Low risk
- 1 = High risk

Patient Data:
- Age: {age} years
- Blood Pressure: {systolic_bp}/{diastolic_bp} mmHg
- Blood Sugar: {bs} mmol/L
- Body Temperature: {body_temp}°{temp_unit}
- Heart Rate: {heart_rate} bpm
- Account Type: {account_type}
- Model Prediction: {ml_model_output} (Probability: {probability:.2f})
- Feature Importances: {feature_importances}
- Patient History: {patient_history}

Conversation history:
{history}

Current question: {question}

Guidelines for response:
- Base your reasoning only on the ML risk score, the listed patient variables, and established medical best practices.
- Provide actionable, evidence-based recommendations tailored for Kenyan healthcare context.
- Include specific Kenyan food examples (ugali, sukuma wiki, beans, etc.) for nutrition advice.
- Keep responses clear, structured, and medically accurate.
- Always identify yourself as "Afya Jamii AI" when asked.
"""

            self.prompt = PromptTemplate(
                input_variables=[
                    "age", "systolic_bp", "diastolic_bp", "bs", "body_temp", 
                    "temp_unit", "heart_rate", "account_type", "ml_model_output", 
                    "probability", "feature_importances", "patient_history", 
                    "history", "question"
                ],
                template=template
            )
            
            # Initialize memory
            self.memory = ConversationBufferMemory(
                memory_key="history",
                input_key="question",
                return_messages=True
            )
            
            # Create chain
            self.chain = LLMChain(
                llm=self.llm,
                prompt=self.prompt,
                memory=self.memory,
                verbose=settings.DEBUG
            )
            
            logger.info("Groq LLM initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self.llm = None
    
    def generate_advice(self, prompt_data: dict) -> str:
        """Generate clinical advice using Groq LLM"""
        if not self.chain:
            return "LLM service temporarily unavailable. Please try again later."
        
        try:
            response = self.chain.run(**prompt_data)
            return response
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return f"Error generating advice: {str(e)}"
    
    def clear_memory(self):
        """Clear conversation memory"""
        if self.memory:
            self.memory.clear()

# Global LLM instance
afya_llm = AfyaJamiiLLM()

def initialize_llm_service():
    """Initialize LLM service on application startup"""
    return afya_llm.llm is not None