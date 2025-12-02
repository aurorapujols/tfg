import datetime
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

from preprocessing import *

from pathlib import Path

import scipy.cluster.hierarchy as sch

if __name__ == "__main__":

    # Open dataset
    dataset_filepath = "../../../data/upftfg26/apujols/processed/dataset_FINAL_20251124-02.csv"
    dataset = pd.read_csv(dataset_filepath, sep=";")

    # Process datetime
    dataset["date_str"] = dataset["filename"].str.split("_").str[0]
    dataset["year"] = dataset["date_str"].str[1:5].astype('int')
    dataset["month"] = dataset["date_str"].str[5:7].astype('int')
    dataset["day"] = dataset["date_str"].str[7:9].astype('int')
    dataset[['year', 'month', 'day', 'hour', 'minute']] = dataset[['year', 'month', 'day', 'hour', 'minute']].astype(int)
    dataset['datetime'] = pd.to_datetime(dataset[['year', 'month', 'day', 'hour', 'minute']], errors='coerce')
    
    # Fill nulls
    dataset['class'] = dataset['class'].fillna("unknown")

    # Get with and height of sum_image_cropeed
    filenames = list(dataset['filename'])#['M20251005_195119_MasLaRoca_NE'] # list(dataset['filename'])
    not_found_counter = 0
    for filename in filenames:
        try:
            filepath = f"../../../data/upftfg26/apujols/processed/sum_image_cropped/{filename}_CROP_SUMIMG.png"
            img = cv2.imread(filepath)
            height, width, _ = img.shape
            dataset.loc[dataset['filename'] == filename, 'height'] = height
            dataset.loc[dataset['filename'] == filename, 'width'] = width
            print(f"{width}x{height}")
        except:
            print(f"Didn't find image for {filename}")
            not_found_counter += 1

    print("NOT FOUND: ", not_found_counter)