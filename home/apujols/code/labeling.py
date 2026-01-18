import os
import glob
import cv2
import shutil
import py7zr
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

from pathlib import Path

from image_preprocessing import update_dataset


def get_filenames_to_label(input_folder):
    # Extract all .7z archives
    zip_files = glob.glob(f"{input_folder}/*.7z")
    for zf in zip_files:
        with py7zr.SevenZipFile(zf, mode='r') as archive:
            path = Path(input_folder).resolve()
            archive.extractall(path=path)
        os.remove(zf)

    # Collect only the .avi and .xml files, delete the rest
    video_paths = [str(p) for p in Path(input_folder).rglob("*.avi")]
    video_files = [Path(p).stem for p in video_paths]

    return video_files, video_paths

def label_as(dataframe, filenames_list, filepath_list, label):

    print(f"Files to label: {len(filenames_list)}")
    files_to_process = []

    for (file, path) in zip(filenames_list, filepath_list):        
        rows = dataframe.loc[dataframe['filename'] == file]
        if rows.shape[0] == 0:
            print(f"⚠️ file {file} is not in the dataframe!")

            # Process it if possible
            files_to_process.append(Path(path))

        elif rows.shape[0] > 1:
            print(f"⚠️ The file {file} appears twice in the dataframe.")
        else:
            dataframe.loc[dataframe['filename'] == file, 'class'] = label

    return dataframe, files_to_process

def clear_incoming_folder(path_incoming_folder, files_to_leave):
    
    # Get the stems of the files to leave
    files_to_leave_names = [f.stem for f in files_to_leave]

    for f in Path(path_incoming_folder).rglob("*"):  # recursively go through all files/folders
        if f.is_file():
            if f.stem not in files_to_leave_names:
                f.unlink()  # delete the file
        elif f.is_dir():
            # delete empty directories
            try:
                f.rmdir()  # only deletes if empty
            except OSError:
                pass  # directory not empty, leave it    # delete folder recursively


if __name__ == "__main__":

    incoming_folder = "../../../data/upftfg26/apujols/incoming"
    output_folder = "../../../data/upftfg26/apujols/processed"
    raw_data_folder = "../../../data/upftfg26/apujols/raw"
    csv_data_filename = "dataset.csv"
    csv_data_path = f"{output_folder}/{csv_data_filename}"

    LABEL = "meteor"

    if os.path.exists(csv_data_path):
        dataset = pd.read_csv(csv_data_path, sep=";")

        # Remove duplicates just in case there still are some
        dataset = dataset.drop_duplicates(subset = ['filename'], keep='last')

        filenames, filepaths = get_filenames_to_label(input_folder=incoming_folder)
        dataset, files_to_process = label_as(dataframe=dataset, filenames_list=filenames, filepath_list=filepaths, label=LABEL)

        print(f"Files to process: {len(files_to_process)}")
        clear_incoming_folder(path_incoming_folder=incoming_folder, files_to_leave=files_to_process)

        # Process missing files
        dataset = update_dataset(dataset=dataset,
                                 input_folders=[incoming_folder, incoming_folder],
                                 output_folder=output_folder, 
                                 are_meteors=(True if LABEL == "meteor" else False))

        clear_incoming_folder(path_incoming_folder=incoming_folder, files_to_leave=[])
        
        dataset = dataset.drop_duplicates(subset = ['filename'], keep='last')
        print(f"unique={len(list(dataset['filename'].unique()))}")
        print(f"num_rows={dataset.shape[0]}")

        dataset.to_csv(csv_data_path, sep=";", index=False)

    else:
        print("⚠️ CSV doesn't exist!")

    print("DONE!")