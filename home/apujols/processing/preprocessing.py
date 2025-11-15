import os
import glob
import cv2
import shutil
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

from pathlib import Path

def folder_exists(folder_path):
    if os.path.exists(folder_path):
        print(f"✅ Folder exists: {folder_path}")
        return True
    else:
        print(f"❌ Folder does NOT exist: {folder_path}")
        return False
    
def has_videos(folder_path):

    if folder_exists(folder_path):
        
        return any(Path(folder_path).rglob("*.avi"))
    
    return None

def apply_mask(frame, mask):
  """
    ``apply_mask`` applies a the binary mask `mask` to the given `frame` of a video.

    :param frame: frame of a video in `cv2.COLOR_BGR2GRAY` format
    :param mask: binary mask with same size as `frame`
    :return: returns the frame image with the mask applied
    """ 
  return cv2.bitwise_and(frame, frame, mask=mask)

def print_bounding_box(img_bgr, bbox):

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Get size of bounding box
    width = bbox['x_max'] - bbox['x_min']
    height = bbox['y_max'] - bbox['y_min']

    # Plot image with bounding box
    fig, ax = plt.subplots()
    ax.imshow(img_rgb)
    bounding_box = patches.Rectangle((bbox['x_min'], bbox['y_min']), width, height, linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(bounding_box)
    plt.title("Image with bounding box")
    plt.axis('off')
    plt.show()

def get_bbox_metadata(input_path, padding=32):

    # Open XML element
    tree = ET.parse(input_path)
    root = tree.getroot()

    # Get video data
    width = float(root.attrib['cx'])
    height = float(root.attrib['cy'])
    frames = float(root.attrib['frames'])
    fps = float(root.attrib['fps'])

    # Get event occurrance time
    year = float(root.attrib['y'])
    month = float(root.attrib['m'])
    day = float(root.attrib['d'])
    hour = float(root.attrib['h'])
    minute = float(root.attrib['m'])

    # Get location data
    lng = float(root.attrib['lng'])
    lat = float(root.attrib['lat'])
    alt = float(root.attrib['alt'])
    camera = str(root.attrib['sid'])

    # Get all uc_path elements
    path_points = root.find('ufocapture_paths')

    x_vals = [float(p.attrib['x']) for p in path_points.findall('uc_path')]
    y_vals = [float(p.attrib['y']) for p in path_points.findall('uc_path')]
    brightnes_vals = [float(p.attrib['bmax']) for p in path_points.findall('uc_path')]
    frame_nums = [float(p.attrib['fno']) for p in path_points.findall('uc_path')]

    # Compute bounding box
    bbox = {
        'x_min': max(0, min(x_vals) - padding),
        'x_max': min(width, max(x_vals) + padding),
        'y_min': max(0, min(y_vals) - padding),
        'y_max': min(height, max(y_vals) + padding)
    }

    # Compute time and brightness
    time_seconds = (frame_nums[-1] - frame_nums[0] + 1) / fps
    mean_brightness = np.mean(brightnes_vals)

    metadata = {
        "filename": Path(input_path).stem,
        "year": year,
        "month": month,
        "day": day,
        "hour": hour,
        "minute": minute,
        "lng": lng,
        "lat": lat,
        "alt": alt,
        "camera": camera,
        "width": width,
        "height": height,
        "frames": frames,
        "fps": fps,
        "time": time_seconds,
        "mean_brightness": mean_brightness
    }

    return bbox, metadata

def generate_sum_image(img_input_path, xml_input_path, output_path, mask_type="folgueroles"):

    # STEP 1: Read the video -----------------------------------------------------------------
    cap = cv2.VideoCapture(img_input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {img_input_path}")
    
    # ---- Read the properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # STEP 2: Define a mask ------------------------------------------------------------------
    my_mask = np.zeros((height, width), dtype=np.uint8)
    if mask_type == "folgueroles":
        mask_height = height - 30
        cv2.rectangle(my_mask, (0, 0), (width, mask_height), 255, thickness=-1)
    elif mask_type == "none":
        cv2.rectangle(my_mask, (0, 0), (width, height), 255, thickness=-1)
    else:
        raise ValueError("Invalid mask_type")

    # STEP 3: Read first frame and perpare background ----------------------------------------
    ret, first_frame = cap.read()
    if not ret:
        raise IOError("Cannot read first frame")
    
    first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    masked_first_frame = apply_mask(first_frame_gray, my_mask)

    # STEP 4: Process frames -----------------------------------------------------------------
    frame_index = 0

    # Sum-images and (optional) stacks initialization
    sum_image = np.zeros_like(masked_first_frame, dtype=np.uint8)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1    # Increment frame that we are processing

        current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    # Grayscaled first frame
        current_frame_masked = apply_mask(current_frame_gray, my_mask)
        frame_diff = cv2.absdiff(current_frame_masked, masked_first_frame)    # Subtract first frame to remove background
        sum_image = np.maximum(sum_image, frame_diff)     # Add frame to the sum-image (getting the max value per pixel)

    out_path = Path(output_path) / f"{Path(img_input_path).stem}_SUMIMG.png"
    cv2.imwrite(str(out_path), sum_image)

    cap.release()

    return sum_image

def generate_cropped_sum_image(sum_img, img_input_path, xml_input_path, output_path, padding=64):

    # Get bounding box and crop sum-image
    bbox, metadata = get_bbox_metadata(input_path=xml_input_path, padding=padding)

    x_min, x_max = int(bbox['x_min']), int(bbox['x_max'])
    y_min, y_max = int(bbox['y_min']), int(bbox['y_max'])

    cropped_sum_img = sum_img[y_min:y_max, x_min:x_max]

    out_path = Path(output_path) / f"{Path(img_input_path).stem}_CROP_SUMIMG.png"
    cv2.imwrite(str(out_path), cropped_sum_img)

    return cropped_sum_img, bbox, metadata

def get_xml_from_video(video_path, xml_paths):

    for xml_path in xml_paths:
        if Path(xml_path).stem == Path(video_path).stem:
            return xml_path
    return None

def preprocess_files(video_files, xml_files, output_folder):

    new_samples_data = []
    skipped = []
    
    # Preprocess them
    N = len(video_files)
    for idx in range(1,N+1):

        if idx%100 == 0:
            print(f"Processed {idx}/{N}")

        curr_video_file = video_files[idx-1]
        curr_xml_file = get_xml_from_video(curr_video_file, xml_files)

        if curr_xml_file is None:
            skipped.append(Path(curr_video_file).stem)
            print(f"⚠️ No XML found for {curr_video_file}, skipping.")
            continue
        
        sum_image = generate_sum_image(img_input_path=curr_video_file, xml_input_path=curr_xml_file, output_path=f"{output_folder}/sum_image")
        sum_image_cropped, bbox, metadata = generate_cropped_sum_image(sum_img=sum_image,
                                                                        img_input_path=curr_video_file, 
                                                                        xml_input_path=curr_xml_file, 
                                                                        output_path=f"{output_folder}/sum_image_cropped")
    
    new_samples_data.append(metadata)

    print(f"Finished: processed {N}/{N} videos")

    return new_samples_data, skipped

def update_dataset(dataset, input_folders, output_folder):
    """
    `update_dataset` checks if there are any new_samples in the new_samples folder and processes them and adds them in the given dataframe
    """
    # Look at the add_samples folder to see if there are new files
    if has_videos(input_folders[0]):
        
        # Get all the files
        video_files = glob.glob(f"{input_folders[0]}/*.avi")
        xml_files = glob.glob(f"{input_folders[1]}/*.xml")

        new_samples_data, skipped = preprocess_files(video_files, xml_files, output_folder)
        
        df_new = pd.DataFrame(new_samples_data)
        dataset = pd.concat([dataset, df_new])

        dataset = dataset[~dataset.index.duplicated(keep="first")]

        # Move the incoming data into the processed raw data folder only if its new data
        if input_folders[0] == incoming_folder:
            raw_xmls_folder = Path(f"{raw_data_folder}/metadata")
            raw_videos_folder = Path(f"{raw_data_folder}/videos")

            # Move XMLs
            for xml_path in xml_files and Path(xml_path).stem not in skipped:
                src = Path(xml_path)
                dst = raw_xmls_folder / src.name
                shutil.move(str(src), str(dst))

            for video_path in video_files and Path(video_path).stem not in skipped:
                src = Path(video_path)
                dst = raw_videos_folder / src.name
                shutil.move(str(src), str(dst))

    return dataset

def create_dataset(raw_data_folder, output_folder):

    dataset = pd.DataFrame()
    
    dataset = update_dataset(dataset, input_folders=[f"{raw_data_folder}/videos", f"{raw_data_folder}/metadata"], output_folder=output_folder)

    return dataset

if __name__ == "__main__":

    incoming_folder = "../../../data/upftfg26/apujols/incoming"
    output_folder = "../../../data/upftfg26/apujols/processed"
    raw_data_folder = "../../../data/upftfg26/apujols/raw"
    csv_data_filename = "dataset.csv"
    csv_data_path = f"{output_folder}/{csv_data_filename}"

    if os.path.exists(csv_data_path):
        temp_df = pd.read_csv(csv_data_path, sep=";")
    else:
        temp_df = create_dataset(raw_data_folder=raw_data_folder, output_folder=output_folder)

    df = update_dataset(dataset=temp_df, input_folders=[incoming_folder, incoming_folder], output_folder=output_folder)

    df.to_csv(csv_data_path, sep=";", index=False)

    print("DONE!")