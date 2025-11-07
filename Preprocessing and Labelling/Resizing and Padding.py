# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 22:09:44 2025

@author: ampat
"""

import cv2
import os

# --- SETTINGS ---
input_folder = r"C:\Users\ampat\Desktop\DS203\Project\DS203-2025-S1-E7-project-images\images"   # folder containing original images
output_folder = r"C:\Users\ampat\Desktop\DS203\Project\Preprocessed images"   # folder where padded images will be saved
target_size = (800, 600)            # (width, height)

# --- CREATE OUTPUT FOLDER IF NOT EXISTS ---
os.makedirs(output_folder, exist_ok=True)

def resize_with_padding(img, target_w, target_h):
    h, w = img.shape[:2]
    aspect = w / h
    target_aspect = target_w / target_h

    # scale image while keeping aspect ratio
    if w < target_w and h < target_h:
        resized = img  # no scaling
        new_w, new_h = w, h
    else:
        if aspect > target_aspect:
            new_w = target_w
            new_h = int(target_w / aspect)
        else:
            new_h = target_h
            new_w = int(target_h * aspect)
            
    resized = cv2.resize(img, (new_w, new_h))
    
    # create black canvas (or white by using 255 instead of 0)
    padded = 255 * np.ones((target_h, target_w, 3), dtype=np.uint8)

    # position the resized image at the center
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return padded

import numpy as np

# --- PROCESS EACH IMAGE IN FOLDER ---
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif")):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"⚠️ Could not read file: {filename}")
            continue
        
        padded_img = resize_with_padding(img, *target_size)
        
        out_path = os.path.join(output_folder, filename)
        cv2.imwrite(out_path, padded_img)
        print(f"✅ Saved: {out_path}")

print("All images processed!")