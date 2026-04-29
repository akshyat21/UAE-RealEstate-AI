# predict_and_explain.py
import os
import json
import joblib
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

# --- Load model and features ---
model = joblib.load('uae_price_predictor_best.pkl')
with open('features.json', 'r') as f:
    required_features = json.load(f)

# --- LLM Explanation Setup ---
class PropertyAnalysis(BaseModel):
    predicted_price_aed: int = Field(description="Price predicted by CatBoost model")
    explanation: str = Field(description="2-3 sentence explanation of key factors influencing price")
    confidence: float = Field(description="Model confidence (R² score)")

parser = PydanticOutputParser(pydantic_object=PropertyAnalysis)

prompt = PromptTemplate(
    template="""You are a UAE real estate pricing expert.
Given property features: {features}
The CatBoost model predicts a price of AED {price}.
Provide a 2-3 sentence explanation focusing on location, property type, size, and any distance-based factors (e.g., proximity to Burj Khalifa, metro, beach).
{format_instructions}""",
    input_variables=["features", "price"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
chain = prompt | llm | parser

def predict_and_explain(input_dict):
    """input_dict: dictionary with feature values (keys must match required_features)"""
    # Create DataFrame with correct columns, fill missing with 0
    input_df = pd.DataFrame([input_dict])[required_features]
    price_pred = int(model.predict(input_df)[0])
    # Format readable feature string for LLM
    features_str = ", ".join([f"{k}: {v}" for k, v in input_dict.items() if v != 0 and 'dist_to' in k or k in ['beds','baths']])
    result = chain.invoke({"features": features_str, "price": price_pred})
    return result.predicted_price_aed, result.explanation