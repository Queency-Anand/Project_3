# import streamlit as st
# import torch.nn.functional as F
# from PIL import Image
# from transformers import AutoModelForTokenClassification, AutoTokenizer
# from transformers import LayoutLMv2Processor
# import torch
# import easyocr
# import cv2
# import os

# # OCR function
# def perform_ocr(image_path):
#     # Initialize EasyOCR reader
#     reader = easyocr.Reader(['en'])  # Change 'en' to the language you're using
#     print(image_path)
#     # # Read image
#     img = cv2.imread("/Users/qanand/Documents/DocBank/DocBank_samples/DocBank_samples/"+image_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
#     # Perform OCR
#     result = reader.readtext(img)
    
#     # Extract words and bounding boxes
#     words = []
#     boxes = []
#     for detection in result:
#         word = detection[1]
#         box = detection[0]  # Bounding box coordinates already in the required format
#         for w in word.split():  # Split text into words
#             words.append(w)
#             boxes.append(box)
    
#     return words, boxes

    

# # Load the model and tokenizer
# model_name = "microsoft/layoutlmv3-base"
# model = AutoModelForTokenClassification.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # Define the prediction function
# def predict_labels(image_path):
#     # Perform OCR to get words and bounding boxes
#     words, boxes = perform_ocr(image_path)
#     print(words)
#     print(boxes)
#     # processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")

#     # img = Image.open("/Users/qanand/Documents/DocBank/DocBank_samples/DocBank_samples/"+image_path).convert("RGB")
#     # # Encode the inputs
#     # encoding = processor(img, words, boxes=boxes,
#     #                      return_tensors="pt", truncation=True, padding="max_length")
    
#     # # Extract input tensors
#     # input_ids = encoding['input_ids']
#     # bbox = encoding['bbox']
#     # attention_mask = encoding['attention_mask']
#     # image = encoding['image'].float()
    
#     # # Perform inference
#     # with torch.no_grad():
#     #     outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, pixel_values=image)
    
#     # # Process the outputs
#     # logits = outputs.logits
#     # probabilities = F.softmax(logits, dim=-1)
#     # predicted_labels = probabilities.argmax(dim=-1)
#     # predictions = predicted_labels.cpu().numpy()
    
#     # return predictions
    

# # Streamlit app
# st.title("DocBank Token Classification")

# # File uploader for the image
# uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# if uploaded_file is not None:
#     # Display the uploaded image
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)
#     # Get the temporary path of the uploaded file
#     image_path = uploaded_file.name  # You can customize the path as needed
#     st.write(image_path)
#     with open(image_path, "wb") as f:
#         f.write(uploaded_file.read())

#     # Button to trigger prediction
#     if st.button("Predict"):
#         # Perform prediction
#         predictions = predict_labels(image_path)
#         st.write("Predicted Labels:", predictions)

import streamlit as st
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import LayoutLMv2Processor
from transformers import LayoutLMv3ForTokenClassification
import torch
import easyocr
import cv2
from transformers import AutoProcessor

# OCR function
# def generate_bounding_box(coordinates):
#     # Extract top-left and bottom-right coordinates
#     xmin, ymin = coordinates[0]
#     xmax, ymax = coordinates[2]

import multiprocessing
import argparse
import pdfplumber
import os
from tqdm import tqdm
from pdfminer.layout import LTChar, LTLine
import re
from collections import Counter
import pdf2image
from pdf2image import convert_from_path
import numpy as np
from PIL import Image
from nanonets import NANONETSOCR
import ast
def within_bbox(bbox_bound, bbox_in):
    assert bbox_bound[0] <= bbox_bound[2]
    assert bbox_bound[1] <= bbox_bound[3]
    assert bbox_in[0] <= bbox_in[2]
    assert bbox_in[1] <= bbox_in[3]

    x_left = max(bbox_bound[0], bbox_in[0])
    y_top = max(bbox_bound[1], bbox_in[1])
    x_right = min(bbox_bound[2], bbox_in[2])
    y_bottom = min(bbox_bound[3], bbox_in[3])

    if x_right < x_left or y_bottom < y_top:
        return False

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox_in_area = (bbox_in[2] - bbox_in[0]) * (bbox_in[3] - bbox_in[1])

    if bbox_in_area == 0:
        return False

    iou = intersection_area / float(bbox_in_area)

    return iou > 0.95


def worker(image_path):
    tokens = []
    boxes = []

    model_1 = NANONETSOCR()
    model_1.set_token('80e415e0-fe2a-11ee-95b2-0af213468460')
    words = model_1.convert_to_boxes(image_path)[0]
   
    for word in words:
        word_bbox = (word['xmin']), (word['ymin']), (word['xmax']), (word['ymax'])
        
        # Preprocess word_bbox
        width = int(word_bbox[2] - word_bbox[0])
        height = int(word_bbox[3] - word_bbox[1])
        
        f_x0 = min(1000, max(0, int(word_bbox[0] / width * 1000)))
        f_y0 = min(1000, max(0, int(word_bbox[1] / height * 1000)))
        f_x1 = min(1000, max(0, int(word_bbox[2] / width * 1000)))
        f_y1 = min(1000, max(0, int(word_bbox[3] / height * 1000)))
        
        word_bbox = (f_x0, f_y0, f_x1, f_y1)
        
        word_bbox = [int(t) for t in word_bbox]  # Convert to integers
        
        # Append token and processed bounding box to lists
        word_text = re.sub(r"\s+", "", word['text'])
        tokens.append(word_text)
        boxes.append(word_bbox)
    print(boxes)
    print(tokens)  
    return tokens, boxes

   

# Load the model and tokenizer
model_name = "microsoft/layoutlmv3-base"
model = LayoutLMv3ForTokenClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def predict_labels(image_path):
    # Perform OCR to get words and bounding boxes
    words, boxes = worker(image_path)

    processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    image =  Image.open("/Users/qanand/Documents/DocBank/DocBank_samples/Docbank_dataset/"+image_path).convert("RGB")
    # Encode the inputs
    encoding = processor(image, words, boxes=boxes,
                         return_tensors="pt", truncation=True, padding="max_length")
    
    # Extract input tensors
    input_ids = encoding['input_ids']
    bbox = encoding['bbox'][0]
    attention_mask = encoding['attention_mask'][0]
    pixel_values = encoding['pixel_values'].float()[0]
   
    # Perform inference
    # Put the model in evaluation mode
    model.eval()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, pixel_values=pixel_values)
    
    # Process the outputs
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=-1)
    predicted_labels = probabilities.argmax(dim=-1)
    predictions = predicted_labels.cpu().numpy()
    
    return predictions

# Streamlit app
st.title("DocBank Token Classification")

# File uploader for the image
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Get the temporary path of the uploaded file
    image_path = "uploaded_images/" + uploaded_file.name  # You can customize the path as needed
    image.save(image_path)

    # Button to trigger prediction
    if st.button("Predict"):
        # Perform prediction
        predictions = predict_labels(image_path)
        st.write("Predicted Labels:", predictions)
