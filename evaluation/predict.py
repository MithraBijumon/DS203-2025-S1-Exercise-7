import cv2
import pandas as pd
import numpy as np
import os
import glob
import pickle
from skimage.feature import local_binary_pattern
from pulp import LpProblem, LpMaximize, LpVariable, value, PULP_CBC_CMD
from tqdm import tqdm

#IMPORTANT Note: This code must include the extract_features_for_cell() function which has been defined in the embedding/feature_extractor.py file

# --- CONFIGURATION ---
MODEL_FILE = "wildlife_model.pkl"         # Your saved .pkl model file
INPUT_IMAGE_DIR = "preprocessed_images"     # Folder of new, unseen images
OUTPUT_IMAGE_DIR = "predicted_images"     # Folder for highlighted images
OUTPUT_CSV_FILE = "predictions.csv"       # The final CSV output

# --- Grid & Preprocessing Constants ---
TARGET_WIDTH = 800
TARGET_HEIGHT = 600
TARGET_ASPECT_RATIO = TARGET_WIDTH / TARGET_HEIGHT
GRID_ROWS = 8
GRID_COLS = 8
CELL_HEIGHT = TARGET_HEIGHT // GRID_ROWS # 75
CELL_WIDTH = TARGET_WIDTH // GRID_COLS   # 100

# --- MAIN PREDICTION SCRIPT ---
if __name__ == "__main__":
    
    # --- 1. Load Model ---
    print(f"Loading model from {MODEL_FILE}...")
    if not os.path.exists(MODEL_FILE):
        print(f"Error: Model file {MODEL_FILE} not found.")
        print("Please run model_trainer.py first.")
        exit()
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully.")

    # --- 2. Setup Directories ---
    if not os.path.exists(INPUT_IMAGE_DIR):
        print(f"Error: Input directory {INPUT_IMAGE_DIR} not found.")
        print("Please create it and add images you want to predict.")
        exit()
        
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
    print(f"Output images will be saved to {OUTPUT_IMAGE_DIR}")
    
    image_paths = glob.glob(os.path.join(INPUT_IMAGE_DIR, '*.jpg'))
    image_paths += glob.glob(os.path.join(INPUT_IMAGE_DIR, '*.png'))
    image_paths += glob.glob(os.path.join(INPUT_IMAGE_DIR, '*.jpeg'))
    
    if not image_paths:
        print(f"Error: No images found in {INPUT_IMAGE_DIR}.")
        exit()

    # --- 3. Run Prediction Loop ---
    all_csv_rows = []
    
    for image_path in tqdm(image_paths, desc="Processing images"):
        image_name = os.path.basename(image_path)
        
        # Load and preprocess the image
        processed_image = cv2.imread(image_path)
        if processed_image is None:
            print(f"Warning: Skipping invalid image {image_name}")
            continue
        
        # This copy is for drawing our highlights on
        output_image = processed_image.copy()
        
        # This list will hold 'c01', 'c02', ... predictions
        csv_row = [image_name]
        
        # Iterate over the 8x8 grid
        for i in range(GRID_ROWS):
            for j in range(GRID_COLS):
                # Extract the 100x75 cell
                y1, y2 = i * CELL_HEIGHT, (i + 1) * CELL_HEIGHT
                x1, x2 = j * CELL_WIDTH, (j + 1) * CELL_WIDTH
                cell = processed_image[y1:y2, x1:x2]
                
                # --- CORE ML PIPELINE ---
                # 1. Extract features
                features = extract_features_for_cell(cell)
                
                # 2. Reshape features AND convert to a named DataFrame
                # We convert the NumPy array to a DataFrame with the correct column names
                features_df = pd.DataFrame(features.reshape(1, -1), columns=FEATURE_NAMES)
                
                # 3. Predict (0 or 1) using the DataFrame
                prediction = model.predict(features_df)[0]
                # ---
                
                # Add prediction to our CSV row
                csv_row.append(int(prediction))
                
                # --- 4. Visualize Prediction ---
                if prediction == 1:
                    # Create a semi-transparent green overlay
                    overlay = output_image.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1) # Draw a filled green box
                    
                    alpha = 0.4 # Transparency
                    cv2.addWeighted(overlay, alpha, output_image, 1 - alpha, 0, output_image)
                    
        # Add this image's row of 64 predictions to our list
        all_csv_rows.append(csv_row)
        
        # Save the highlighted image
        output_path = os.path.join(OUTPUT_IMAGE_DIR, f"pred_{image_name}")
        cv2.imwrite(output_path, output_image)

    # --- 4. Save Final CSV Output ---
    print("\nSaving final predictions CSV...")
    
    # Create column headers: 'image_name', 'c01', 'c02', ..., 'c64'
    headers = ['ImageFileName'] + [f'c{i:02d}' for i in range(1, (GRID_ROWS * GRID_COLS) + 1)]
    
    # Create and save the DataFrame
    csv_df = pd.DataFrame(all_csv_rows, columns=headers)
    csv_df.to_csv(OUTPUT_CSV_FILE, index=False)
    
    print(f"Prediction complete. View highlighted images in '{OUTPUT_IMAGE_DIR}' and results in '{OUTPUT_CSV_FILE}'.")
