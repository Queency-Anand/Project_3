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
    words = model_1.convert_to_boxes(image_path)

    width = 0 
    height = 0

    with Image.open(image_path) as img:
        width, height = img.size
    
    for word in words:
        word_bbox = int(word['xmin']), int(word['ymin']), int(word['xmax']), int(word['ymax'])

        # print("Image Width:", width)
        # print("Image Height:", height)

        if width == 0 or height == 0:
            continue

        f_x0 = min(999, max(0, int(word_bbox[0] / width * 1000)))
        f_y0 = min(999, max(0, int(word_bbox[1] / height * 1000)))
        f_x1 = min(999, max(0, int(word_bbox[2] / width * 1000)))
        f_y1 = min(999, max(0, int(word_bbox[3] / height * 1000)))

        # print("WORD",word)

        # print(word_bbox[0], width, word_bbox[0] / width, word_bbox[0] / width * 1000)
        # print(word_bbox[1], height, word_bbox[1] / height, word_bbox[1] / height * 1000)
        word_bbox = (f_x0, f_y0, f_x1, f_y1)
        
        word_bbox = [int(t) for t in word_bbox]  # Convert to integers
        
        # Append token and processed bounding box to lists
        word_text = re.sub(r"\s+", "", word['text'])
        tokens.append(word_text)
        boxes.append(word_bbox)
      print(boxes)
      print(tokens)  
    return tokens, boxes
