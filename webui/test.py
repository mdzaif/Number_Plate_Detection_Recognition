import torch
import torch_directml
import cv2
import numpy as np
import gradio as gr
import easyocr
import os
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import random
from PIL import Image

# Device configuration
device = torch_directml.device() if torch_directml.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# YOLO Model Loading
yolo_model_path = r"C:\\Users\\Administrator\\deployment\\weights\\best.pt"
if not os.path.exists(yolo_model_path):
    raise FileNotFoundError(f"YOLO model file not found: {yolo_model_path}")

model = YOLO(yolo_model_path)
model.fuse()

# Super-Resolution Model Loading
sr_model_path = r"C:\\Users\\Administrator\\deployment\\TF-ESPCN\\export\\ESPCN_x2.pb"
if not os.path.exists(sr_model_path):
    raise FileNotFoundError(f"Super-Resolution model file not found: {sr_model_path}")

sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(sr_model_path)
sr.setModel("espcn", 2)

# OCR Reader
reader = easyocr.Reader(['bn'], gpu=torch.cuda.is_available())

# Image Processing Functions
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
# Video Processing with OCR
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Error: Could not open video."

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output paths
    video_output_path = output_dir / (Path(video_path).stem + '_output.mp4')
    csv_output_path = output_dir / (Path(video_path).stem + '_results.csv')

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(video_output_path), fourcc, fps, (width, height))

    # Data collection for CSV
    ocr_data = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = frame.copy()

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cropped_image = frame[y1:y2, x1:x2]
            
            # Get OCR results
            text_upscale, text_upscale_morph = extract_text(cropped_image)
            
            # Store data for CSV
            ocr_data.append({
                "frame": frame_count,
                "bbox": f"{x1},{y1},{x2},{y2}",
                "upscale_text": text_upscale,
                "upscale_morph_text": text_upscale_morph
            })
            
            # Draw bounding box and text
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, text_upscale, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(annotated_frame)
        frame_count += 1

    # Save CSV
    df = pd.DataFrame(ocr_data)
    df.to_csv(csv_output_path, index=False)

    cap.release()
    out.release()

    return str(video_output_path), str(csv_output_path)

# Updated Gradio Interface
def gradio_interface(img, vid):
    if img is not None:
        processed_image, ocr_text, ocr_text_morph = process_image(img)
        return processed_image, ocr_text, ocr_text_morph, None, None
    
    if vid is not None:
        video_path, csv_path = process_video(vid.name)
        return None, None, None, video_path, csv_path
    
    return None, None, None, None, None

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
        gr.File(label="Processed Video"),
        gr.File(label="OCR Results CSV")
    ]
)

if __name__ == "__main__":
    iface.launch()