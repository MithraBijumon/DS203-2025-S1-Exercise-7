import cv2
import pandas as pd
import numpy as np
import os
from skimage.feature import local_binary_pattern
from tqdm import tqdm # A progress bar! pip install tqdm

# --- CONFIGURATION ---
PREPROCESSED_DIR = "preprocessed_images"
LABELS_FILE = "label_results.csv" # The manually created labels (Should be in the same folder as this python file)
OUTPUT_DATASET = "training_dataset_baseline.csv" # The file this script will create

GRID_ROWS = 8
GRID_COLS = 8
CELL_HEIGHT = 600 // GRID_ROWS # 75 pixels
CELL_WIDTH = 800 // GRID_COLS  # 100 pixels

# LBP (Local Binary Patterns) configuration
LBP_POINTS = 24 # Number of points to check around a pixel
LBP_RADIUS = 3  # Radius of the circle

def extract_features_for_cell(cell):
    """
    This function takes one 100x75 cell (a BGR image) and returns
    a 1D numpy array of its calculated baseline features.
    """
    features = []

    # --- 1. Color Features (Average & Std Dev BGR) ---
    # We now capture both mean and standard deviation
    (means, stds) = cv2.meanStdDev(cell)
    features.extend(means.flatten()) # Add B_mean, G_mean, R_mean
    features.extend(stds.flatten()) # Add B_std, G_std, R_std

    # --- 2. Grayscale & Edge Features ---
    gray_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    
    # Canny Edge Density
    # We still need grayscale stats to find good Canny thresholds
    (gray_mean, gray_std) = cv2.meanStdDev(gray_cell)
    sigma = gray_std[0][0]
    v = gray_mean[0][0]
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    
    edges = cv2.Canny(gray_cell, lower, upper)
    edge_density = np.sum(edges > 0) / (CELL_WIDTH * CELL_HEIGHT)
    features.append(edge_density)

    # --- 3. Texture Features (LBP) ---
    # We use the 'uniform' method, which is robust and limits features.
    # It results in (LBP_POINTS + 2) features.
    lbp = local_binary_pattern(gray_cell, LBP_POINTS, LBP_RADIUS, method="uniform")
    
    # Create a normalized histogram of the LBP results
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, LBP_POINTS + 3),
                             range=(0, LBP_POINTS + 2))
    
    # Normalize histogram to be a probability distribution
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6) # Add epsilon to avoid division by zero
    
    features.extend(hist)
    
    return np.array(features)

# --- MAIN SCRIPT ---
if __name__ == "__main__":
    print("Starting feature extraction...")
    
    # Load the labels
    if not os.path.exists(LABELS_FILE):
        print(f"Error: Labels file not found at {LABELS_FILE}")
        exit()
    labels_df = pd.read_csv(LABELS_FILE)
    
    all_features = []
    all_labels = []

    # Use tqdm for a progress bar. Iterating rows is like iterating images.
    for index, row in tqdm(labels_df.iterrows(), total=labels_df.shape[0], desc="Processing Images"):
        image_name = row['image_name']
        image_path = os.path.join(PREPROCESSED_DIR, image_name)
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not load image {image_name}. Skipping.")
            continue
            
        # Iterate over the 8x8 grid
        for i in range(GRID_ROWS):
            for j in range(GRID_COLS):
                # Calculate coordinates for the cell
                y1, y2 = i * CELL_HEIGHT, (i + 1) * CELL_HEIGHT
                x1, x2 = j * CELL_WIDTH, (j + 1) * CELL_WIDTH
                
                # Extract the 100x75 cell
                cell = image[y1:y2, x1:x2]
                
                # Get the label for this cell from the CSV
                # Column name is 'c01', 'c02', ... 'c64'
                cell_label_col = f'c{(i * GRID_COLS + j + 1):02d}'
                label = row[cell_label_col]
                
                # --- This is the core step ---
                features = extract_features_for_cell(cell)
                
                all_features.append(features)
                all_labels.append(label)

    print("\nFeature extraction complete. Assembling final dataset...")

    # Create a list of feature names for the CSV header
    # This list is now updated to match our baseline features
    feature_names = ['B_mean', 'G_mean', 'R_mean', 'B_std', 'G_std', 'R_std', 'Edge_density']
    # Add LBP histogram feature names
    feature_names += [f'LBP_bin_{k}' for k in range(LBP_POINTS + 2)]

    # Create the final DataFrame
    X = np.array(all_features)
    y = np.array(all_labels)
    
    dataset_df = pd.DataFrame(X, columns=feature_names)
    dataset_df['label'] = y
    
    # Save to CSV
    dataset_df.to_csv(OUTPUT_DATASET, index=False)
    
    print(f"Success! Training dataset saved to {OUTPUT_DATASET}")
    print(f"Total rows (cells): {len(dataset_df)}")
    print(f"Total features: {len(feature_names)}")
