import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class AfyaJamiiLLM:
    def __init__(self):
        self.llm = None
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
You are Afya Jamii AI, a clinical decision-support and maternal nutrition assistant for Kenyan pregnant and postnatal mothers and general users seeking nutrition advice.

This is the context for the current conversation:
{context}

This is the conversation history:
{history}

Based on the context and history, answer the following question:
Question: {question}

Guidelines for response:
- If patient data is available in the context or history, base your reasoning on it.
- Provide actionable, evidence-based recommendations tailored for Kenyan healthcare context.
- Include specific Kenyan food examples (ugali, sukuma wiki, beans, etc.) for nutrition advice.
- Keep responses clear, structured, and medically accurate.
- Do not mention the underlying ML model unless asked.
- Always identify yourself as "Afya Jamii AI" when asked.
"""

            self.prompt = PromptTemplate(
                input_variables=["context", "history", "question"],
                template=template
            )
            
            # Create chain
            self.chain = LLMChain(
                llm=self.llm,
                prompt=self.prompt,
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

# Global LLM instance
afya_llm = AfyaJamiiLLM()

def initialize_llm_service():
    """Initialize LLM service on application startup"""
    return afya_llm.llm is not None