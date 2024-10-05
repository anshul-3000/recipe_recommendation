from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import re
import ast
import os

app = Flask(__name__)

# Load vocabulary and DataFrame
MODEL_PATH = 'D:\\Recipe Recommendation - api-main\\models\\recipe_recommendation_model.pkl'
DATA_PATH = 'D:\\Recipe Recommendation - api-main\\data\\processed\\recipes_processed.csv'

# Check if the model and data paths exist
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file not found at {DATA_PATH}")

# Load the model and data
with open(MODEL_PATH, 'rb') as f:
    vocab = pickle.load(f)

df = pd.read_csv(DATA_PATH)

# Convert 'Ingredient_Vector' from string to list
df['Ingredient_Vector'] = df['Ingredient_Vector'].apply(ast.literal_eval)

# Function to vectorize ingredients
def vectorize_ingredients(ingredient_str, vocab):
    vector = [0] * len(vocab)
    ingredients = [ingredient.strip().lower() for ingredient in re.split(r',|\s', ingredient_str) if ingredient]
    for ingredient in ingredients:
        if ingredient in vocab:
            vector[vocab.index(ingredient)] = 1
    return vector

# Function to recommend recipes
def recommend_recipes(user_ingredient_str, df, top_n=5):
    user_vector = vectorize_ingredients(user_ingredient_str, vocab)
    recipe_vectors = np.array(df['Ingredient_Vector'].tolist())
    user_vector = np.array(user_vector).reshape(1, -1)

    # Compute cosine similarity between user vector and recipe vectors
    similarity_scores = cosine_similarity(user_vector, recipe_vectors)[0]
    
    # Get top N recipe indices based on similarity scores
    recipe_similarity = list(enumerate(similarity_scores))
    top_indices = sorted(recipe_similarity, key=lambda x: x[1], reverse=True)[:top_n]

    return [index for index, score in top_indices]

@app.route('/api/recommend', methods=['POST'])
def recommend():
    # Get ingredients from POST request
    user_ingredients = request.json.get('ingredients')
    if not user_ingredients:
        return jsonify({'error': 'No ingredients provided.'}), 400
    
    try:
        top_recipes = recommend_recipes(user_ingredients, df, top_n=5)
        results = []

        for recipe_idx in top_recipes:
            name = df.loc[recipe_idx, 'Title']
            ingredients = df.loc[recipe_idx, 'Core_Ingredients']
            image_url = df.loc[recipe_idx, 'Image Link']
            
            # Format ingredients nicely
            formatted_ingredients = ', '.join(ingredients.split())
            
            results.append({
                'name': name,
                'ingredients': formatted_ingredients,
                'image_url': image_url
            })

        return jsonify({'recommended_recipes': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
