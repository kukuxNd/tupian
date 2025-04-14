# -*- coding: utf-8 -*-
import os
from rapidocr_onnxruntime import RapidOCR
import imageio.v2 as imageio
from PIL import Image
import cv2
import numpy as np
import csv
import pyglet

def dds_to_png(dds_path):
    try:
        # Load DDS file using pyglet
        image = imageio.imread(dds_path, format='DDS')
        # ��ͼ������ת��ΪPillowͼ�����
        return image
    except Exception as e:
        print(f"Error loading {dds_path}: {str(e)}")
        return None

def scan_dds_files(root_dir):
    ocr_engine = RapidOCR()
    results = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.dds'):
                file_path = os.path.join(root, file)
                
                # Convert DDS to numpy array
                image = dds_to_png(file_path)

                if image is None:
                    continue
                    
                # Get image dimensions
                height, width = image.shape[:2]
                
                # Convert to PIL format
                #img_pil = Image.fromarray(img)
                pil_image = Image.fromarray(image)
                # Perform OCR
                ocr_result, _ = ocr_engine(pil_image)
                
                if ocr_result:
                    # Get first character's coordinates and text
                    first_char = ocr_result[0]
                    x, y = first_char[0][0]  # Top-left coordinate of first character
                    text = first_char[1]     # Recognized text
                    
                    # Format output
                    # Store the result as a dictionary
                    result = {
                        "file_path": file_path,
                        "width": width,
                        "height": height,
                        "x": int(x),
                        "y": int(y),
                        "text": text
                    }
                    results.append(result)
                    # ��ӡ��ǰ���
                    print(f"file_path={file_path}, width={width}, height={height}, x={int(x)}, y={int(y)}, text={text}")
    
    
    return results

if __name__ == "__main__":
    dds_dir = "data/"
    results = scan_dds_files(dds_dir)
    
    # Write results to a text file
    with open("results.csv", "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["file_path", "width", "height", "x", "y", "text"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()  # Write the header
        for result in results:
            writer.writerow(result)
    
    print(f"Results saved to results.csv")
