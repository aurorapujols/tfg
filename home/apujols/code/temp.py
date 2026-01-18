import py7zr
import xml.etree.ElementTree as ET
import pandas as pd
import shutil
import tempfile

from pathlib import Path

def set_min_max_brightness(dataset, filename, Bmin, Bmax):
    """
    Function to set 'bmin' and 'bmax' of the row with filename given in filepath and in the given dataset.
    """
    dataset.loc[dataset['filename'] == filename, 'bmin'] = Bmin
    dataset.loc[dataset['filename'] == filename, 'bmax'] = Bmax
    
    return dataset

def extract_only_xml_to_temp(archive_path):
    """
    Extract only XML files from a .7z archive into a temporary directory.
    Returns the Path to the temporary folder.
    """
    temp_dir = Path(tempfile.mkdtemp())

    with py7zr.SevenZipFile(archive_path, mode='r') as archive:
        # List all files inside the archive
        all_files = archive.getnames()

        # Filter only XML files
        xml_files = [f for f in all_files if f.lower().endswith(".xml")]

        # Extract only XML files
        archive.extract(targets=xml_files, path=temp_dir)

    return temp_dir

def process_xml_folder(folder_path):
    """
    Process XML files inside a folder and compute bmin/bmax for each.
    Returns {filename_stem: (bmin, bmax)}.
    """
    folder = Path(folder_path)
    brightness = {}

    for xml_file in folder.rglob("*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        path_points = root.find('ufocapture_paths')
        bright_vals = [float(p.attrib['bmax']) for p in path_points.findall('uc_path')]

        bmin = min(bright_vals)
        bmax = max(bright_vals)

        brightness[xml_file.stem] = (int(bmin), int(bmax))

    return brightness

if __name__ == "__main__":

    # ----------------------------------------------------
    # Merge datasets for all files to have bmin and bmax
    # ----------------------------------------------------
    """
        df1 = pd.read_csv("../../../data/upftfg26/apujols/processed/dataset_fixed.csv", sep=";")
        df2 = pd.read_csv("../../../data/upftfg26/apujols/processed/dataset_temp.csv", sep=";")

        df_merged = df1.merge( df2[["filename", "bmin", "bmax"]], on="filename", how="left", suffixes=("", "_from_df2") )
        df_merged["bmin"] = df_merged["bmin"].fillna(df_merged["bmin_from_df2"]) 
        df_merged["bmax"] = df_merged["bmax"].fillna(df_merged["bmax_from_df2"])

        df_merged = df_merged.drop(columns=["bmin_from_df2", "bmax_from_df2"])
        df_merged.to_csv("../../../data/upftfg26/apujols/processed/dataset_merged.csv", sep=";", index=False)
    """

    # ----------------------------------------------------
    # Correct the date fields in a dataset & add "unknown" label
    # ----------------------------------------------------
    """
        df = pd.read_csv("../../../data/upftfg26/apujols/processed/dataset.csv", sep=";")
        print(f"Dataset shape: {df.shape}")

        # MYYYYMMDD_XXXXXX_MasLaRoca_XX
        df["year"] = df["filename"].str.slice(1, 5) # YYYY
        df["month"] = df["filename"].str.slice(5, 7) # MM
        df["day"] = df["filename"].str.slice(7, 9) # DD
        df["class"] = df["class"].where(df["class"] == "meteor", "unknown")

        df.to_csv("../../../data/upftfg26/apujols/processed/dataset_fixed.csv", sep=";", index=False)
        print("Saved dataset_fixed.csv")
    """


    # ----------------------------------------------------
    # Add `bmin` and `bmax` to files that are missing it
    # ----------------------------------------------------
    """
        # 1. Extract only XML files to a temporary folder
        temp_folder = extract_only_xml_to_temp("../../../data/upftfg26/apujols/raw_ALL_20251206_M_202511.7z")

        # 2. Process XMLs from disk
        brightness_data = process_xml_folder(temp_folder)

        # 3. Load dataset
        dataset = pd.read_csv("../../../data/upftfg26/apujols/processed/dataset_temp.csv", sep=";")

        # 4. Update dataset
        for name, (bmin, bmax) in brightness_data.items():
            dataset = set_min_max_brightness(dataset, name, Bmin=bmin, Bmax=bmax)

        # 5. Save updated dataset
        dataset.to_csv("../../../data/upftfg26/apujols/processed/dataset_temp.csv", sep=";", index=False)

        # 6. Clean up temp folder
        shutil.rmtree(temp_folder)
    """

    # ----------------------------------------------------
    # Move all files that appear in a dataset to folder
    # ----------------------------------------------------
    """
        dataset = pd.read_csv("../../../data/upftfg26/apujols/processed/dataset_temp.csv", sep=";")
        dst_folder = Path("../../../data/upftfg26/apujols/processed/original")
        count = 0
        missed = 0
        for filename in dataset["filename"]:
            src = Path(f"../../../data/upftfg26/apujols/processed/sum_image_cropped/{filename}_CROP_SUMIMG.png")
            if src.exists():
                shutil.move(str(src), str(dst_folder / src.name))
                count += 1
            else:
                missed += 1
        print(f"Count={count}   Missed={missed}.")
    """
