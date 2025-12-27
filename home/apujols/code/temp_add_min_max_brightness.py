import py7zr
import xml.etree.ElementTree as ET
import pandas as pd

from pathlib import Path

def set_min_max_brightness(dataset, filename, Bmin, Bmax):
    """
    Function to set 'bmin' and 'bmax' of the row with filename given in filepath and in the given dataset.
    """
    dataset.loc[dataset['filename'] == filename, 'bmin'] = Bmin
    dataset.loc[dataset['filename'] == filename, 'bmax'] = Bmax
    
    return dataset

import py7zr

def load_xml_from_7z_in_memory(archive_path):
    """
    Load XML files from a .7z archive directly into memory.
    Returns a dict: {xml_filename: xml_string}
    """
    xml_data = {}

    with py7zr.SevenZipFile(archive_path, mode='r') as archive:
        # read() returns a dict: {filename: BytesIO}
        extracted = archive.readall()

        for name, bio in extracted.items():
            if name.lower().endswith(".xml"):
                xml_data[Path(name).stem] = bio.read().decode("utf-8")

    return xml_data


def process_xml_data(xml_data):

    meteors_brightness = {}
    for name, xml_string in xml_data.items():

        tree = ET.ElementTree(ET.fromstring(xml_string))
        root = tree.getroot()

        path_points = root.find('ufocapture_paths')
        brightnes_vals = [float(p.attrib['bmax']) for p in path_points.findall('uc_path')]
        bmin = min(brightnes_vals)
        bmax = max(brightnes_vals)

        meteors_brightness[name] = (int(bmin), int(bmax))

    return meteors_brightness


if __name__ == "__main__":

    xml_data = load_xml_from_7z_in_memory("data/upftfg26/apujols/metadata.7z")

    brightness_data = process_xml_data(xml_data)

    dataset = pd.read_csv("data/upftfg26/apujols/processed/dataset_new.csv", sep=";")

    for name, tpl_brightness in brightness_data.items():
        dataset = set_min_max_brightness(dataset, name, Bmin=tpl_brightness[0], Bmax=tpl_brightness[1])
    
    dataset.to_csv("data/upftfg26/apujols/processed/dataset.csv", sep=";", index=False)