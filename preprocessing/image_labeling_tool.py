import cv2
import numpy as np
import pandas as pd
import os

# --- CONFIGURATION ---
# Directory where your 800x600 preprocessed images are stored
IMAGE_DIRECTORY = "preprocessed_images"
# The name of the output CSV file
OUTPUT_CSV_FILE = "labels.csv"     #IMPORTANT: Name this as "labels_{your_name}.csv"
# Grid dimensions
GRID_ROWS = 8
GRID_COLS = 8

# --- HELPER FUNCTIONS ---
def get_image_files(directory):
    """Returns a sorted list of image files from a directory."""
    files = []
    for f in os.listdir(directory):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            files.append(os.path.join(directory, f))
    return sorted(files)

def initialize_labels(image_files, csv_path):
    """Initializes a DataFrame for labels, loading existing data if available."""
    if os.path.exists(csv_path):
        print(f"Loading existing labels from {csv_path}...")
        df = pd.read_csv(csv_path)
        # Ensure image_name is string
        df['image_name'] = df['image_name'].astype(str)
        # Add any new images that are not in the CSV yet
        existing_images = set(df['image_name'])
        new_images = [os.path.basename(f) for f in image_files if os.path.basename(f) not in existing_images]
        if new_images:
            print(f"Found {len(new_images)} new images to label.")
            new_df = pd.DataFrame({
                'image_name': new_images,
                **{f'c{i:02d}': 0 for i in range(1, GRID_ROWS * GRID_COLS + 1)}
            })
            df = pd.concat([df, new_df], ignore_index=True)
    else:
        print("No existing labels file found. Creating a new one.")
        image_basenames = [os.path.basename(f) for f in image_files]
        df = pd.DataFrame({
            'image_name': image_basenames,
            **{f'c{i:02d}': 0 for i in range(1, GRID_ROWS * GRID_COLS + 1)}
        })
    # Set image_name as the index for easy lookup
    df.set_index('image_name', inplace=True)
    return df

def draw_grid_and_labels(image, labels_for_image):
    """Draws the grid and highlights labeled cells on the image."""
    display_image = image.copy()
    h, w, _ = display_image.shape
    cell_h, cell_w = h // GRID_ROWS, w // GRID_COLS

    for i in range(GRID_ROWS):
        for j in range(GRID_COLS):
            # Draw grid lines
            start_point = (j * cell_w, i * cell_h)
            end_point = ((j + 1) * cell_w, (i + 1) * cell_h)
            cv2.rectangle(display_image, start_point, end_point, (0, 255, 255), 1)

            # Check for label and highlight if it's 1
            cell_index = i * GRID_COLS + j + 1
            if labels_for_image[f'c{cell_index:02d}'] == 1:
                overlay = display_image.copy()
                cv2.rectangle(overlay, start_point, end_point, (0, 255, 0), -1)
                alpha = 0.4  # Transparency factor
                display_image = cv2.addWeighted(overlay, alpha, display_image, 1 - alpha, 0)
    return display_image

def mouse_callback(event, x, y, flags, param):
    """Handles mouse clicks to toggle labels."""
    global labels_df, current_image_basename, needs_redraw

    if event == cv2.EVENT_LBUTTONDOWN:
        # Calculate which grid cell was clicked
        h, w = 600, 800  # Assuming 800x600, but should be from image shape
        cell_h, cell_w = h // GRID_ROWS, w // GRID_COLS
        row = y // cell_h
        col = x // cell_w
        cell_index = row * GRID_COLS + col + 1
        label_col = f'c{cell_index:02d}'

        # Toggle the label (0 to 1, 1 to 0)
        current_label = labels_df.loc[current_image_basename, label_col]
        labels_df.loc[current_image_basename, label_col] = 1 - current_label
        print(f"Toggled {current_image_basename} - Cell {cell_index} to {1 - current_label}")
        needs_redraw = True


# --- MAIN SCRIPT ---
if not os.path.exists(IMAGE_DIRECTORY):
    print(f"Error: The directory '{IMAGE_DIRECTORY}' does not exist.")
    print("Please create it and place your preprocessed 800x600 images inside.")
    exit()

image_files = get_image_files(IMAGE_DIRECTORY)
if not image_files:
    print(f"Error: No images found in '{IMAGE_DIRECTORY}'.")
    exit()

labels_df = initialize_labels(image_files, OUTPUT_CSV_FILE)

current_image_index = 0
current_image_basename = ""
needs_redraw = True

cv2.namedWindow("Image Labeler")
cv2.setMouseCallback("Image Labeler", mouse_callback)

print("\n--- Controls ---")
print("Click on a cell to toggle its label (wildlife or not).")
print("'n' -> Next Image")
print("'p' -> Previous Image")
print("'s' -> Save labels to CSV")
print("'q' -> Quit")
print("----------------\n")


while True:
    current_image_path = image_files[current_image_index]
    current_image_basename = os.path.basename(current_image_path)

    # Load the original image
    image = cv2.imread(current_image_path)
    if image is None or image.shape[0] != 600 or image.shape[1] != 800:
        print(f"Warning: Skipping invalid image or wrong dimensions: {current_image_path}")
        # Move to next image automatically if current one is bad
        if current_image_index < len(image_files) -1:
            current_image_index +=1
            needs_redraw = True
            continue
        else:
            break
        #image = cv2.resize(image, (800, 600))

    # Get labels for the current image
    labels_for_current_image = labels_df.loc[current_image_basename]

    # Draw the grid and highlights
    display_image = draw_grid_and_labels(image, labels_for_current_image)

    # Add status text
    status_text = f"Image {current_image_index + 1}/{len(image_files)}: {current_image_basename}"
    cv2.putText(display_image, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(display_image, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Image Labeler", display_image)
    needs_redraw = False

    key = cv2.waitKey(1) & 0xFF

    # Handle keyboard input
    if key == ord('q'):
        break
    elif key == ord('n'):
        if current_image_index < len(image_files) - 1:
            current_image_index += 1
            needs_redraw = True
    elif key == ord('p'):
        if current_image_index > 0:
            current_image_index -= 1
            needs_redraw = True
    elif key == ord('s'):
        # Reset index before saving to make 'image_name' a column again
        labels_df.reset_index().to_csv(OUTPUT_CSV_FILE, index=False)
        print(f"\nLabels saved to {OUTPUT_CSV_FILE}!\n")

cv2.destroyAllWindows()
print("Labeling session finished.")
