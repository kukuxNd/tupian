#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
完整流程说明：
1. 使用 RapidOCR 识别图片中的中文文字及其位置（汉字检测）。
2. 根据检测到的文字区域，利用 inpainting 算法将文字区域修复（文字抹除）。
3. 使用 googletrans 将检测到的中文文本翻译为英文（若翻译出错则保留原文本）。
4. 在修复后的图片上以相似排版和风格绘制翻译后的英文文本，输出合成后的图片。
 
注意：
- 请安装依赖：
    pip install opencv-python pillow rapidocr-onnxruntime googletrans==4.0.0-rc1
- 本示例用 RapidOCR 做文字检测，检测结果中每条数据应包含 'text'（识别文本）、'score'（置信度）以及 'box'（检测区域的四个点，[x,y] 坐标）。
- 部分参数（如置信度阈值、字体大小、inpainting 半径等）可根据实际图片调整。
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rapidocr_onnxruntime import RapidOCR
from googletrans import Translator

# 如果需要，在 Windows 下请设置 pytesseract 路径（本例仅用于 inpainting 部分读取图像，无 OCR）
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 配置参数：输入、输出图片路径；字体文件路径（请保证字体文件存在或换成其他可用字体）
INPUT_IMAGE_PATH = 'out_img/F8.png'    # 待处理的图片路径
OUTPUT_IMAGE_PATH = 'output.png'  # 处理后保存的图片路径
FONT_PATH = 'arial.ttf'           # 英文字体路径

def ocr_and_get_boxes(image_path, conf_threshold=0.3):
    """
    使用 RapidOCR 进行汉字检测，返回检测到的文字及其位置（bounding box）。
    检测结果每项包含：
      - 'text': 识别的文本内容
      - 'score': 置信度分数（0～1）
      - 'box': 四个角点坐标，格式为 [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    
    返回值为列表，每个元素为字典：
      { 'text': ..., 'left': x, 'top': y, 'width': w, 'height': h, 'conf': score }
    """
    # 读取图片（由于 RapidOCR 接受 numpy 数组，下面使用 OpenCV 读取，转换 BGR -> RGB）
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法读取图片，请检查图片路径。")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 初始化 OCR 模型
    ocr_model = RapidOCR()
    results = ocr_model(img_rgb)
    
    boxes = []
    # for res in results:
    #     if not isinstance(res, list) or len(res) != 3:
    #         # 若结构异常，可跳过
    #         continue
    res = results[0][0]
    box_points = res[0]  # 四个点坐标
    text = res[1]        # 识别的文本
    score = res[2]       # 置信度

    # 过滤低置信度或空文本
    # if not text.strip():
    #     continue
    # if score < conf_threshold:
    #     continue

    # 计算矩形边界
    xs = [pt[0] for pt in box_points]
    ys = [pt[1] for pt in box_points]
    left = int(min(xs))
    top = int(min(ys))
    right = int(max(xs))
    bottom = int(max(ys))
    width = right - left
    height = bottom - top

    boxes.append({
        'text': text,
        'left': left,
        'top': top,
        'width': width,
        'height': height,
        'conf': score
    })
    return boxes

def remove_text_by_inpainting(image_path, boxes):
    """
    对检测到的文字区域进行 inpainting 文字抹除。
    读取原图，创建与图像尺寸一致的 mask（文字区域设为白色），
    然后调用 cv2.inpaint 修复文字区域，返回修复后图像（BGR 格式）。
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法读取图片，请检查图片路径。")
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for box in boxes:
        left = box['left']
        top = box['top']
        right = left + box['width']
        bottom = top + box['height']
        cv2.rectangle(mask, (left, top), (right, bottom), 255, -1)
    # 使用 INPAINT_TELEA 算法进行修复，半径参数可根据需要调整
    inpainted = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    return inpainted

def translate_chinese_to_english(text_list):
    """
    使用 googletrans 将一组中文文本翻译为英文。
    对每条文本进行翻译（若出错则保留原文本）。
    返回翻译后的英文文本列表。
    """
    translator = Translator()
    results = []
    for txt in text_list:
        if not txt.strip():
            results.append('')
            continue
        try:
            res = translator.translate(txt, src='zh-cn', dest='en')
            results.append(res.text)
        except Exception as e:
            print(f"翻译出错：'{txt}' 错误信息: {e}")
            results.append(txt)
    return results

def paste_text_on_image(opencv_image, boxes, new_texts, font_path=FONT_PATH, font_size=40):
    """
    在图像上绘制翻译后的英文文本。
    1. 将 OpenCV 图像（BGR）转换为 PIL 格式（RGB），以便使用 Pillow 绘图。
    2. 依据每个文字区域，将对应的英文文本居中绘制到区域内。
    返回绘制后的图像（转换回 BGR 格式）。
    """
    pil_image = Image.fromarray(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        print("加载字体失败，使用默认字体。", e)
        font = ImageFont.load_default()
    
    for box, new_text in zip(boxes, new_texts):
        if not new_text.strip():
            continue
        left = box['left']
        top = box['top']
        width = box['width']
        height = box['height']
        # 使用 font.getsize() 获取文本宽度和高度
        bbox = font.getbbox(new_text)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        # 计算文本绘制位置，使其在检测区域中居中
        x = left + (width - text_w) / 2
        y = top + (height - text_h) / 2
        draw.text((x, y), new_text, font=font, fill=(255, 255, 255))
    
    result_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return result_img

def main():
    # 1. 使用 RapidOCR 获取图片中的中文文字及其位置信息
    boxes = ocr_and_get_boxes(INPUT_IMAGE_PATH, conf_threshold=0.3)
    if not boxes:
        print("未检测到可用中文文字或置信度太低，程序结束。")
        return
    print("检测到的文字及位置信息：")
    for box in boxes:
        print(f"文本: '{box['text']}', 置信度: {box['conf']:.2f}")

    # 2. 使用 inpainting 修复文字区域（文字抹除）
    inpainted_img = remove_text_by_inpainting(INPUT_IMAGE_PATH, boxes)

    # 3. 翻译检测到的中文文本为英文
    original_texts = [box['text'] for box in boxes]
    translated_texts = translate_chinese_to_english(original_texts)
    print("翻译结果：")
    for cn, en in zip(original_texts, translated_texts):
        print(f"{cn} -> {en}")

    # 4. 在修复后的图像上绘制翻译后的英文文本（尽可能保持原排版）
    final_img = paste_text_on_image(inpainted_img, boxes, translated_texts, font_path=FONT_PATH, font_size=40)

    # 5. 保存处理后的图片
    cv2.imwrite(OUTPUT_IMAGE_PATH, final_img)
    print(f"处理完成，结果保存在：{OUTPUT_IMAGE_PATH}")

if __name__ == '__main__':
    main()
