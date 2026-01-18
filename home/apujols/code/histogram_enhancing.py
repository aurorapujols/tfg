import os
import cv2 # openCV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

from plotly.subplots import make_subplots
from pathlib import Path

def meteor_stretch(img, Bmin, Bmax):
    if Bmax == Bmin:
        # Binary image: pixels equal to Bmax -> 255, rest -> 0
        out = np.zeros_like(img, dtype=np.uint8)
        out[img == Bmax] = 255
        return out

    stretched = (img - Bmin) * (255.0 / (Bmax - Bmin))
    stretched = np.clip(stretched, 0, 255)
    return stretched.astype(np.uint8)

def global_threshold(img, T):
    return (img >= T).astype(np.uint8) * 255

def min_max_stretch(img): 
    x_min = img.min() 
    x_max = img.max()   
    
    # Avoid division by zero if the image is flat 
    if x_max == x_min: 
        return np.zeros_like(img, dtype=np.uint8) 
    
    stretched = (img - x_min) * (255.0 / (x_max - x_min)) 
    return stretched.astype(np.uint8)

def percentile_stretch(img, low=2, high=98):
    p_low = np.percentile(img, low)
    p_high = np.percentile(img, high)
    stretched = np.clip((img - p_low) * (255.0 / (p_high - p_low)), 0, 255)
    return stretched

def cv2_equalizer(img):
    return cv2.equalizeHist(img)

if __name__ == "__main__": 

    dataset = pd.read_csv(f"../../../data/upftfg26/apujols/processed/dataset_temp.csv", sep=";")
    output_folder = "percentile_stretch"

    # Use only samples we are gonna use for clustering
    """ v1
        subset = dataset[(dataset['month'].isin([10,11])) & (dataset['year'] == 2025)]
        meteors = subset[subset['class'] == 'meteor'] #.sample(n=500, random_state=42)
        unknowns = subset[subset['class'] == 'unknown'] #.sample(n=500, random_state=42)
        subset = pd.concat([meteors, unknowns])
    """

    # To take a balanced subset (but we want to process all of them)
    # meteors = dataset[(dataset['class'] == 'meteor')]    # all meteor samples
    # unknowns = dataset[dataset['class'] != 'meteor']    # all unknown images
    # n_min = min(len(meteors), len(unknowns))
    # meteors_sample = meteors.sample(n=n_min, random_state=42)
    # unknowns_sample = unknowns.sample(n=n_min, random_state=42)
    # subset = pd.concat([meteors_sample, unknowns_sample]).sort_values("filename").reset_index(drop=True)

    subset = dataset.copy()
    print(f"Final dataset shape: {subset.shape}")
    
    for filename in subset['filename']:
        input_path = f"../../../data/upftfg26/apujols/processed/sum_image_cropped/{filename}_CROP_SUMIMG.png"
        output_path = f"../../../data/upftfg26/apujols/processed/{output_folder}/{filename}_CROP_ENHANCED.png"

        if not os.path.exists(output_path):
            img_greyscale_cropped = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            bmin = subset.loc[subset["filename"] == filename, "bmin"].iloc[0]
            bmax = subset.loc[subset["filename"] == filename, "bmax"].iloc[0]
            # enhanced_img = meteor_stretch(img=img_greyscale_cropped, Bmin=bmin, Bmax=bmax)
            # enhanced_img = global_threshold(img=img_greyscale_cropped, T=bmin)
            # enhanced_img = min_max_stretch(img=img_greyscale_cropped)
            enhanced_img = percentile_stretch(img=img_greyscale_cropped, low=2, high=98)
            success = cv2.imwrite(output_path, enhanced_img)