# -*- coding: utf-8 -*-
import os
from rapidocr_onnxruntime import RapidOCR
import imageio.v2 as imageio
from PIL import Image
import cv2
import numpy as np

import pyglet

#测试数量


test_count = 10
def png_to(png_path):
    try:
        # Load DDS file using pyglet
        image = imageio.imread(png_path)
        # 转换为灰度图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 二值化处理
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary
    except Exception as e:
        print(f"Error loading {png_path}: {str(e)}")
        return None

def dds_to_png(dds_path):
    try:
        # Load DDS file using pyglet
        image = imageio.imread(dds_path, format='DDS')
        # 转换为灰度图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 二值化处理
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        #cv2.imshow('Image with Chinese Characters', binary)
        return binary
    except Exception as e:
        print(f"Error loading {dds_path}: {str(e)}")
        return None

def scan_dds_files(root_dir):
    ocr_engine = RapidOCR(
    use_gpu=True,                            # 是否使用 GPU
    gpu_id=0,                                 # GPU ID
    rec_batch_num=6,                          # 识别批处理大小
    det_thresh=0.3,                           # 检测置信度阈值
    rec_thresh=0.9                            # 识别置信度阈值
    )
    results = []
    file_count = sum([len(files) for _, _, files in os.walk(root_dir)])
    print(f"Total files in {root_dir}: {file_count}")
    run_count = 0
    test_count = 10
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            run_count = run_count + 1
            print(f"scan:{run_count}/{file_count}")
            
            #限定测试数量
            if run_count >test_count :
                return
            
            #读dds
            if file.endswith('.dds'):
                # Convert DDS to numpy array
                image = dds_to_png(file_path)
            else:
                #读 png
                image = png_to(file_path)

            cv2.imshow('Image with Chinese Characters', image)
            cv2.waitKey(0)
            #return
                # Get image dimensions
            height, width = image.shape[:2]
            
            # 将图像数据转换为Pillow图像对象
            pil_image = Image.fromarray(image)
            # Perform OCR
            ocr_result, _ = ocr_engine(pil_image)
            
            if ocr_result:
                # Get first character's coordinates and text
                # Initialize x and y with a large value
                x = float('inf')
                y = float('inf')
                
                # Iterate through all elements in ocr_result to find the minimum x and y
                for char in ocr_result:
                    char_x, char_y = char[0][0]  # Top-left coordinate of the character
                    if char_x < x:
                        x = char_x
                    if char_y < y:
                        y = char_y
                
                # Get the recognized text from the first character
                text = ocr_result[0][1]
                
                # Format output
                result = f"{file_path}[w={width},h={height}][*{int(y)},{int(x)}*]{text}"
                print(result)
                results.append(result)
    
    return results

if __name__ == "__main__":
    dds_dir = "data/"
    results = scan_dds_files(dds_dir)
    
    # Write results to a text file
    with open("results.txt", "w", encoding="utf-8") as file:
        for result in results:
            file.write(result + "\n")
