# predict_and_explain.py
import os
import joblib
import json
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

# ---------- Paths ----------
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'uae_price_predictor_best.pkl')
features_path = os.path.join(script_dir, 'features.json')
distances_path = os.path.join(script_dir, 'area_distances.json')

# ---------- Load model and features ----------
model = joblib.load(model_path)
with open(features_path, 'r') as f:
    required_features = json.load(f)

# ---------- Load area distance lookup ----------
with open(distances_path, 'r') as f:
    area_distances = json.load(f)
# Convert string values back to float
for area in area_distances:
    for key in area_distances[area]:
        area_distances[area][key] = float(area_distances[area][key])

# ---------- LLM Setup ----------
class PropertyAnalysis(BaseModel):
    predicted_price_aed: int = Field(description="CatBoost predicted price")
    explanation: str = Field(description="2-3 sentence explanation focusing on location, property type, and nearby attractions")

parser = PydanticOutputParser(pydantic_object=PropertyAnalysis)

prompt = PromptTemplate(
    template="""You are a UAE real estate pricing expert.
Property details: {beds} bedroom(s), {baths} bathroom(s), type: {prop_type}, area: {area_name}, furnishing: {furnishing}, completion: {completion}.
The model predicts a price of AED {price:,}.
Based on the area's prestige, proximity to landmarks (e.g., Burj Khalifa, beach, metro), and property type, provide a 2-3 sentence explanation of the price.

{format_instructions}""",
    input_variables=["beds", "baths", "prop_type", "area_name", "furnishing", "completion", "price"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
chain = prompt | llm | parser

# ---------- Prediction function ----------
def predict_from_input(beds, baths, area_name, prop_type, furnishing, completion):
    # Initialize feature dict with zeros
    input_dict = {feat: 0 for feat in required_features}
    
    # Basic numerical features
    input_dict['beds'] = beds
    input_dict['baths'] = baths
    
    # One-hot encoded features (exact column names must match training)
    # Note: The column names in features.json are like 'type_Apartment', 'area_name_Jumeirah Village Circle (JVC)', etc.
    type_col = f'type_{prop_type}'
    area_col = f'area_name_{area_name}'
    furnishing_col = f'furnishing_{furnishing}'
    completion_col = f'completion_status_{completion}'
    
    if type_col in input_dict:
        input_dict[type_col] = 1
    if area_col in input_dict:
        input_dict[area_col] = 1
    if furnishing_col in input_dict:
        input_dict[furnishing_col] = 1
    if completion_col in input_dict:
        input_dict[completion_col] = 1
    
    # Add distance features for this area (fallback to zeros if area not found)
    area_dist = area_distances.get(area_name, {})
    for col in required_features:
        if col.startswith('dist_to_') and col in area_dist:
            input_dict[col] = area_dist[col]
    
    # Create dataframe and predict
    input_df = pd.DataFrame([input_dict])[required_features]
    pred_price = int(model.predict(input_df)[0])
    
    # Get LLM explanation
    result = chain.invoke({
        "beds": beds,
        "baths": baths,
        "prop_type": prop_type,
        "area_name": area_name,
        "furnishing": furnishing,
        "completion": completion,
        "price": pred_price
    })
    return pred_price, result.explanation