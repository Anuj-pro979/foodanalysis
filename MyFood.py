# Advanced Food Analyzer
# This application uses Google's Gemini AI with a structured multi-stage analysis process
# to analyze food images and estimate ingredients, quantities, and calories

import streamlit as st
import google.generativeai as genai
import os
import requests
import base64
import tempfile
from PIL import Image
import io
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json

# Configure Google Gemini API with the provided key - hardcoded
API_KEY = "AIzaSyDpaOZq0jE6d4SdTpf1GyNk_lLkB75Kn_8"
genai.configure(api_key=API_KEY)

@dataclass
class IngredientData:
    """Class for storing ingredient analysis data"""
    name: str
    quantity: str
    calories: float
    notes: Optional[str] = None

@dataclass
class AnalysisResult:
    """Class for storing the complete analysis result"""
    ingredients: List[IngredientData]
    total_calories: float
    confidence_score: float
    notes: Optional[str] = None

class FoodAnalyzer:
    """Multi-stage food analysis using Gemini AI"""
    
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        """Initialize the food analyzer with the specified model"""
        self.model = genai.GenerativeModel(model_name)
        
    def _call_gemini_with_image(self, prompt: str, image_data: bytes) -> str:
        """Helper method to call the Gemini API with direct image data"""
        try:
            # Convert image bytes to base64 for Gemini
            image_parts = [
                {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(image_data).decode('utf-8')
                }
            ]
            
            # Generate content with both text prompt and image
            response = self.model.generate_content([prompt, *image_parts])
            return response.text
        except Exception as e:
            return f"Error calling Gemini API: {str(e)}"
    
    def stage1_identify_ingredients(self, image_data: bytes) -> List[str]:
        """Stage 1: Identify all visible ingredients in the food image"""
        prompt = """
        You are a culinary expert specializing in food identification.
        
        Examine the food in the attached image.
        
        TASK: Identify all visible ingredients in this dish.
        
        Instructions:
        - List ONLY the ingredients you can visually confirm
        - Be specific (e.g., "chicken breast" rather than just "meat")
        - Include visible garnishes, sauces, and seasonings
        - Format your response as a JSON array of strings
        
        Example output format:
        ["ingredient1", "ingredient2", "ingredient3"]
        """
        
        response = self._call_gemini_with_image(prompt, image_data)
        
        try:
            # Extract JSON array from response if needed
            if "[" in response and "]" in response:
                json_str = response[response.find("["):response.rfind("]")+1]
                return json.loads(json_str)
            return json.loads(response)
        except:
            # Fallback: extract ingredients line by line if JSON parsing fails
            ingredients = []
            for line in response.split("\n"):
                line = line.strip()
                if line and not line.startswith("#") and not line.startswith("-"):
                    # Clean up any bullet points or numbers
                    clean_line = line.lstrip("•-*0123456789. ")
                    if clean_line:
                        ingredients.append(clean_line)
            return ingredients if ingredients else [response]
    
    def stage2_estimate_quantities(self, ingredients: List[str], image_data: bytes) -> Dict[str, str]:
        """Stage 2: Estimate quantities for each identified ingredient"""
        ingredients_str = ", ".join(ingredients)
        
        prompt = f"""
        You are a culinary measurement specialist.
        
        Examine the food in the attached image.
        
        The following ingredients have been identified: {ingredients_str}
        
        TASK: Estimate reasonable quantities for each ingredient.
        
        Instructions:
        - Provide quantity estimates for EACH ingredient listed
        - Use standard measurements (grams, cups, tablespoons, etc.)
        - Consider visible portion sizes and typical recipe amounts
        - Format your response as a JSON object with ingredients as keys and quantities as values
        
        Example output format:
        {{
          "ingredient1": "100g",
          "ingredient2": "2 tbsp",
          "ingredient3": "1/4 cup"
        }}
        """
        
        response = self._call_gemini_with_image(prompt, image_data)
        
        try:
            # Extract JSON object from response if needed
            if "{" in response and "}" in response:
                json_str = response[response.find("{"):response.rfind("}")+1]
                return json.loads(json_str)
            return json.loads(response)
        except:
            # Fallback: create dictionary manually if JSON parsing fails
            quantities = {}
            current_ingredient = None
            
            for line in response.split("\n"):
                line = line.strip()
                if line:
                    if ":" in line:
                        parts = line.split(":", 1)
                        ing = parts[0].strip().strip('"\'- ')
                        qty = parts[1].strip().strip('"\'- ')
                        quantities[ing] = qty
            
            # If still no quantities, return default values
            if not quantities:
                return {ing: "unknown quantity" for ing in ingredients}
                
            return quantities
    
    def stage3_calculate_calories(self, ingredients_with_quantities: Dict[str, str], image_data: bytes) -> List[Dict[str, Any]]:
        """Stage 3: Calculate calories for each ingredient based on quantities"""
        ingredients_json = json.dumps(ingredients_with_quantities)
        
        prompt = f"""
        You are a nutritional expert specializing in calorie calculation.
        
        The following ingredients and quantities have been identified in the dish shown in the attached image:
        {ingredients_json}
        
        TASK: Calculate estimated calories for each ingredient based on the quantities.
        
        Instructions:
        - Calculate approximate calories for EACH ingredient based on its quantity
        - Use standard nutritional data for your calculations
        - Show your calculation logic for each ingredient
        - Format your response as a JSON array of objects with the following structure:
          [
            {{
              "name": "ingredient name",
              "quantity": "quantity",
              "calories": estimated_calories_as_number,
              "calculation": "brief explanation of calculation"
            }},
            ...
          ]
        """
        
        response = self._call_gemini_with_image(prompt, image_data)
        
        try:
            # Extract JSON array from response if needed
            if "[" in response and "]" in response:
                json_str = response[response.find("["):response.rfind("]")+1]
                return json.loads(json_str)
            return json.loads(response)
        except:
            # Fallback: create summary if JSON parsing fails
            fallback_results = []
            
            for ingredient, quantity in ingredients_with_quantities.items():
                fallback_results.append({
                    "name": ingredient,
                    "quantity": quantity,
                    "calories": 0,  # Can't determine without proper calculation
                    "calculation": "Calorie calculation failed"
                })
                
            return fallback_results
    
    def analyze_food(self, image_data: bytes) -> AnalysisResult:
        """Complete multi-stage analysis of food in image"""
        
        # Stage 1: Identify ingredients
        st.markdown("### Stage 1: Identifying ingredients...")
        ingredients = self.stage1_identify_ingredients(image_data)
        st.success(f"✅ Identified {len(ingredients)} ingredients")
        st.write(ingredients)
        
        # Stage 2: Estimate quantities
        st.markdown("### Stage 2: Estimating quantities...")
        quantities = self.stage2_estimate_quantities(ingredients, image_data)
        st.success("✅ Estimated quantities for ingredients")
        st.write(quantities)
        
        # Stage 3: Calculate calories
        st.markdown("### Stage 3: Calculating calories...")
        calorie_data = self.stage3_calculate_calories(quantities, image_data)
        st.success("✅ Calculated calories for ingredients")
        
        # Process results into structured format
        ingredient_objects = []
        total_calories = 0
        
        for item in calorie_data:
            if isinstance(item, dict):
                calories = item.get("calories", 0)
                if isinstance(calories, str):
                    try:
                        calories = float(calories.replace(',', ''))
                    except:
                        calories = 0
                        
                ingredient = IngredientData(
                    name=item.get("name", "Unknown"),
                    quantity=item.get("quantity", "Unknown"),
                    calories=calories,
                    notes=item.get("calculation", None)
                )
                ingredient_objects.append(ingredient)
                total_calories += calories
        
        # Create final analysis result
        result = AnalysisResult(
            ingredients=ingredient_objects,
            total_calories=total_calories,
            confidence_score=0.85,  # Placeholder confidence score
            notes="Analysis complete across all three stages"
        )
        
        return result

# Function to format the results as a nice table
def format_analysis_results(result: AnalysisResult):
    st.markdown("## 📊 Analysis Results")
    
    # Create table for ingredients
    st.markdown("### Ingredients & Calories")
    
    # Display as table
    ingredients_data = []
    for ing in result.ingredients:
        ingredients_data.append({
            "Ingredient": ing.name,
            "Quantity": ing.quantity,
            "Calories": f"{ing.calories:.1f}" if ing.calories else "N/A",
            "Notes": ing.notes[:50] + "..." if ing.notes and len(ing.notes) > 50 else (ing.notes or "")
        })
    
    st.table(ingredients_data)
    
    # Total calories
    st.markdown(f"### Total Calories: **{result.total_calories:.1f}**")
    
    # Additional notes
    if result.notes:
        st.markdown(f"### Analysis Notes")
        st.markdown(result.notes)

# Main app function
def main():
    st.set_page_config(
        page_title="Advanced Food Analyzer", 
        page_icon="🍽️", 
        layout="wide"
    )
    
    st.title("🍽️ Advanced Food Analyzer")
    st.subheader("Analyze food images to identify ingredients, estimate quantities, and calculate calories")
    
    # Initialize the analyzer with API key already configured
    analyzer = FoodAnalyzer()
    
    # Image input options
    image_option = st.radio("Choose image input method:", 
                         options=["Upload Image", "Take Photo"],
                         horizontal=True)
    
    image_file = None
    if image_option == "Upload Image":
        image_file = st.file_uploader("Upload a food image:", type=["jpg", "jpeg", "png"])
    else:
        image_file = st.camera_input("Take a photo of your food")
    
    if image_file is not None:
        # Display the uploaded/captured image
        col1, col2 = st.columns([1, 2])
        with col1:
            # Fixed the deprecated parameter by using use_container_width instead
            st.image(image_file, caption="Food Image", use_container_width=True)
            
        # Process the image when button is clicked
        if st.button("Analyze Food"):
            # Read image bytes directly without uploading to external service
            image_bytes = image_file.getvalue()
            
            # Analyze the image using our multi-stage analyzer with direct image data
            with st.spinner("Running multi-stage analysis..."):
                with col2:
                    analysis_result = analyzer.analyze_food(image_bytes)
                    format_analysis_results(analysis_result)
    
    # Footer
    st.markdown("---")
    st.caption("This app uses Google's Gemini AI model with structured multi-stage analysis to analyze food images. Results are estimates only.")

if __name__ == "__main__":
    main()
