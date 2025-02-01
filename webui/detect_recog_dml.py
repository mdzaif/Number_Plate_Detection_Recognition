import torch
import torch_directml  # If using DirectML
import cv2
import numpy as np
import gradio as gr
import easyocr
import os
from pathlib import Path
from ultralytics import YOLO
import random
from PIL import Image
import csv

# Device (Use DirectML if available and desired, otherwise "cuda" or "cpu")
device = torch_directml.device() if torch_directml.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
# YOLO Model Loading
yolo_model_path = "./weights/best.pt"  # Replace with your YOLO model path
if not os.path.exists(yolo_model_path):
    raise FileNotFoundError(f"YOLO model file not found: {yolo_model_path}")

model = YOLO(yolo_model_path)
model.fuse()  # Optimize

# Super-Resolution Model Loading
sr_model_path = "./TF-ESPCN/export/ESPCN_x2.pb"  # Replace with your SR model path
if not os.path.exists(sr_model_path):
    raise FileNotFoundError(f"Super-Resolution model file not found: {sr_model_path}")

sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(sr_model_path)
sr.setModel("espcn", 2)

# OCR Reader
reader = easyocr.Reader(['bn'], gpu=torch.cuda.is_available())  # Use GPU if available

# Image Processing Functions (same as before)
def apply_super_resolution(image):
    return sr.upsample(image)

def apply_dilation(image):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

def apply_erosion(image):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.erode(image, kernel, iterations=1)

def preprocess_upscale_only(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return apply_super_resolution(gray_image)

def preprocess_upscale_morph(image):
    upscaled_image = preprocess_upscale_only(image)
    return apply_erosion(apply_dilation(upscaled_image))

def extract_text(image):
    upscaled_image = preprocess_upscale_only(image)
    text_upscale = " ".join(reader.readtext(upscaled_image, detail=0))
    upscaled_morph_image = preprocess_upscale_morph(image)
    text_upscale_morph = " ".join(reader.readtext(upscaled_morph_image, detail=0))
    return text_upscale, text_upscale_morph

# Function to get a random error image from error_img directory
def get_random_error_image():
    error_img_dir = Path("./error_img")
    valid_image_extensions = ['.png', '.jpeg', '.jpg', '.webp']
    images = [f for f in error_img_dir.iterdir() if f.suffix.lower() in valid_image_extensions]
    return random.choice(images) if images else None

# Process Video (With OCR and Save Text to CSV)
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error: Could not open video.", None  # Return error message if video cannot be opened

    # Define output path and properties
    output_dir = Path("outputs")  # Ensure this directory exists for storing videos and text files
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (Path(video_path).stem + '_output.mp4')
    csv_path = output_dir / (Path(video_path).stem + '_ocr_text.csv')

    # Open video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Create CSV file to store OCR results
    with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Bounding Box", "Upscaled OCR Text", "Upscaled + Morph OCR Text"])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Apply YOLO detection to each frame (with OCR)
            results = model(frame)
            annotated_frame = results[0].plot()  # Annotate the frame with detections

            # Extract OCR and save to CSV
            for box in results[0].boxes:  # Iterate through detected objects.
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cropped_image = frame[y1:y2, x1:x2]

                # Extract OCR text
                text_upscale, text_upscale_morph = extract_text(cropped_image)

                # Write the data to CSV
                writer.writerow([f"({x1}, {y1}, {x2}, {y2})", text_upscale, text_upscale_morph])

            out.write(annotated_frame)  # Write annotated frame to output video

    cap.release()
    out.release()

    return str(output_path), str(csv_path)  # Return the full paths of the annotated video and CSV file
# Process Single Image
def process_image(image):
    results = model(image)
    
    # Check if no bounding boxes are found
    if len(results[0].boxes) == 0:
        # No detections, return random error image and custom OCR text
        random_error_image_path = get_random_error_image()
        if random_error_image_path:
            error_image = Image.open(random_error_image_path)
        else:
            error_image = Image.new('RGB', (256, 256), color='red')  # Placeholder image if none found

        return np.array(error_image), "I'm not offense, but the judging cat judge you!", "I'm not offense, but the judging cat judge you!"
    
    annotated_img = results[0].plot()  # Plot detections on the original image.

    ocr_texts = []
    ocr_texts_morph = []
    for box in results[0].boxes:  # Iterate through detected objects.
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cropped_image = image[y1:y2, x1:x2]

        text_upscale, text_upscale_morph = extract_text(cropped_image)
        ocr_texts.append(text_upscale)
        ocr_texts_morph.append(text_upscale_morph)

    combined_ocr_text = " ".join(ocr_texts)
    combined_ocr_text_morph = " ".join(ocr_texts_morph)

    return annotated_img, combined_ocr_text, combined_ocr_text_morph

# Gradio Interface (Updated for Video and Image)
def gradio_interface(img, vid):
    if img is not None:
        # Process image
        processed_image, ocr_text, ocr_text_morph = process_image(img)
        return processed_image, ocr_text, ocr_text_morph, None  # Only image and text outputs
    
    if vid is not None:
        # Process video
        output_video, csv_file = process_video(vid.name)  # Access the video file path properly
        return None, None, None, [output_video, csv_file]  # Video and CSV file output
    
    return None, None, None, None  # In case of empty inputs

iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Image(type='numpy', label='Upload Image'),
        gr.File(label='Upload Video', type="filepath")
    ],
    outputs=[
        gr.Image(label='Processed Image'),
        gr.Textbox(label='OCR Text (Upscaled)'),
        gr.Textbox(label='OCR Text (Upscaled + Morph)'),
        gr.File(label="Download Processed Video and CSV", file_count="multiple")
    ],
    title="Bangla Vehicle Number Plate Detection & Recognition",  # Add title here
    flagging_dir="flagged_data"  # Save flagged inputs/outputs in this folder
)

if __name__ == "__main__":
    iface.launch()
