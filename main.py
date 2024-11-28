import streamlit as st
from helpers.recipe_generator import generate_recipe
from helpers.nutrition_analysis import analyze_nutrition
from helpers.waste_tracker import log_waste, get_waste_tips
from helpers.meal_planner import generate_meal_plan, generate_grocery_list
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import requests
from inference_sdk import InferenceHTTPClient

# Load the saved Indian food classification model
Indian_Food = load_model(r"model files/Indian_food_20.h5")

# Menu options for Indian food
indian_menu = ['burger', 'butter_naan', 'chai', 'chapati', 'chole_bhature', 'dal_makhani', 'dhokla', 'fried_rice', 
               'idli', 'jalebi', 'kaathi_rolls', 'kadai_paneer', 'kulfi', 'masala_dosa', 'momos', 'paani_puri', 
               'pakode', 'pav_bhaji', 'pizza', 'samosa']

# API key for recipe retrieval
API_KEY = '9S+NWmHVRTjvvzwWSpshVA==XWMo7sRyGP3eejiC'

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="Gqf1hrF7jdAh8EsbOoTM"
)

# Food classification and recipe functions
def indian_food_classification(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (228, 228)) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = Indian_Food.predict(img)
    pred_class = np.argmax(prediction, axis=-1)
    return indian_menu[pred_class[0]], np.max(prediction)

def get_recipe(food_class):
    query = food_class.replace('_', ' ')
    api_url = f'https://api.api-ninjas.com/v1/recipe?query={query}'
    response = requests.get(api_url, headers={'X-Api-Key': API_KEY})
    if response.status_code == requests.codes.ok:
        recipes = response.json()
        if recipes:
            recipe_info = recipes[0]  # Assuming the first recipe is the most relevant
            return recipe_info.get('title'), recipe_info.get('ingredients'), recipe_info.get('instructions')
    return None, None, None

def get_nutrition_info(food_class):
    query = food_class.replace('_', ' ')
    api_url = f'https://api.api-ninjas.com/v1/nutrition?query={query}'
    response = requests.get(api_url, headers={'X-Api-Key': API_KEY})
    if response.status_code == requests.codes.ok:
        return response.json()
    return None

def display_recipe(title, ingredients, instructions):
    st.subheader("Recipe:")
    st.write(f"**Title:** {title}")
    st.write("**Ingredients:**")
    st.write(ingredients)
    st.write("**Instructions:**")
    st.write(instructions)

def display_nutrition(nutrition_info):
    st.subheader("Nutrition Information:")
    for nutrient in nutrition_info:
        st.write(f"**{nutrient.get('name', 'Unknown Nutrient')}:**")
        for key, value in nutrient.items():
            if key != 'name':
                st.write(f"- {key.replace('_', ' ').title()}: {value}")
        st.write("")

# Streamlit App
st.set_page_config(page_title="Sustainable Cooking with AI", layout="wide")
st.image('assets/images.jpeg')

st.sidebar.image('assets/images (2).jpeg')

st.sidebar.title("Navigation")
option = st.sidebar.radio(
    "Choose a feature:",
    ["Generative Recipes", "Nutrition Assistance", "Waste Tracker", "Meal Planning", "Food Classification and Recipes"]
)

if option == "Generative Recipes":
    st.header("Generative Recipes")
    st.subheader("Create recipes with available ingredients.")

    ingredients = st.text_input("Enter ingredients (comma-separated):")
    meal_type = st.selectbox("Select meal type:", ["Breakfast", "Lunch", "Dinner"])
    dietary_preference = st.selectbox("Select dietary preference:", ["None", "Vegan", "Gluten-Free", "Keto"])

    if st.button("Generate Recipe"):
        recipe = generate_recipe(ingredients.split(","), meal_type, dietary_preference)
        st.write("### Generated Recipe:")
        st.write(recipe)

elif option == "Nutrition Assistance":
    st.header("Nutrition Assistance")
    st.subheader("Analyze the nutritional content of recipes.")

    ingredients = st.text_input("Enter ingredients for analysis (comma-separated):")
    if st.button("Analyze Nutrition"):
        nutrition_data = analyze_nutrition(ingredients.split(","))
        st.write("### Nutrition Information:")
        for key, value in nutrition_data.items():
            st.write(f"- **{key}:** {value}")

elif option == "Waste Tracker":
    st.header("Waste Tracker")
    st.subheader("Log unused ingredients and reduce food waste.")

    ingredient = st.text_input("Enter ingredient:")
    quantity = st.number_input("Enter quantity:", min_value=0.0, step=0.1)
    use_by_date = st.date_input("Use-by date:")

    if st.button("Log Ingredient"):
        waste_log = log_waste(ingredient, quantity, use_by_date)
        st.write("### Current Waste Log:")
        st.write(waste_log)

    if st.button("Get Waste Tips"):
        tips = get_waste_tips()
        st.write("### Tips to Reduce Waste:")
        for tip in tips:
            st.write(f"- {tip}")

elif option == "Meal Planning":
    st.header("Meal Planning")
    st.subheader("Plan weekly meals and create grocery lists.")

    ingredients = st.text_input("Enter ingredients for meal planning (comma-separated):")
    if st.button("Generate Meal Plan"):
        meal_plan = generate_meal_plan(ingredients.split(","))
        st.write("### Weekly Meal Plan:")
        for day, meals in meal_plan.items():
            st.write(f"**{day}:**")
            st.write(meals)

        grocery_list = generate_grocery_list(meal_plan)
        st.write("### Grocery List:")
        st.write(", ".join(grocery_list))

elif option == "Food Classification and Recipes":
    st.header("Food Classification and Recipes")
    st.subheader("Upload an image to classify food and retrieve recipes.")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")

        if st.button("Classify"):
            prediction, confidence = indian_food_classification(image)
            st.success(f"Predicted Food: {prediction} (Confidence: {confidence:.2f})")

            title, ingredients, instructions = get_recipe(prediction)
            if title:
                display_recipe(title, ingredients, instructions)
                nutrition_info = get_nutrition_info(prediction)
                if nutrition_info:
                    display_nutrition(nutrition_info)
                else:
                    st.warning("Nutrition information not available for this food class.")
            else:
                st.warning("Recipe not found for this food class.")
