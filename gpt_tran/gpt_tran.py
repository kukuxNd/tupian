#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
示例流程：
1. OCR 检测图片中的文字（中文）。
2. 根据 OCR 得到的文字 bounding box，在原图上做 inpainting 去除文字。
3. 将识别到的中文翻译为英文。
4. 在相同位置用尽量类似风格的方式写上英文文字。
5. 保存输出最终图像。
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytesseract
from googletrans import Translator


# ========== 根据需要修改以下路径 ==========
# pytesseract 的可执行文件路径（如果系统 PATH 中已有，则可不设置）
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
# 原始图片文件路径
INPUT_IMAGE_PATH = 'out_img/8S.png'
# 输出结果图片文件路径
OUTPUT_IMAGE_PATH = 'output.png'
# 字体文件路径（用于绘制文字，可替换为系统内其他英文字体）
FONT_PATH = 'Arial.ttf'
# =======================================


def ocr_and_get_boxes(image_path):
    """
    使用 Tesseract OCR 获取图片中的文字及其外接矩形信息。
    返回值为一个列表，列表中的每个元素是 dict：
    {
      'text': 文字内容,
      'left': 边界框左上角 x,
      'top':  边界框左上角 y,
      'width': 边界框宽度,
      'height': 边界框高度,
      'conf': 识别置信度
    }
    """
    # 读取图像（PIL）
    pil_img = Image.open(image_path)

    # 使用 image_to_data 获取每个识别行/词的位置信息
    data = pytesseract.image_to_data(pil_img, lang='chi_sim', output_type=pytesseract.Output.DICT)

    boxes = []
    n_boxes = len(data['level'])
    for i in range(n_boxes):
        text = data['text'][i].strip()
        conf = int(data['conf'][i])
        if text and conf > 30:  # 过滤空字符串和置信度较低的识别结果
            left = data['left'][i]
            top = data['top'][i]
            width = data['width'][i]
            height = data['height'][i]

            boxes.append({
                'text': text,
                'left': left,
                'top': top,
                'width': width,
                'height': height,
                'conf': conf
            })
    return boxes


def remove_text_by_inpainting(image_path, boxes):
    """
    对给定的原图使用 inpainting 方法去除指定位置的文字。
    返回去除文字后的图像（OpenCV 格式 BGR）。
    """
    # 使用 OpenCV 读取原图
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # 创建掩码 mask：与原图同尺寸，文字区域标记为 255，其他区域为 0
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for box in boxes:
        left = box['left']
        top = box['top']
        right = left + box['width']
        bottom = top + box['height']
        # 在 mask 上将文字区域标记为白色
        cv2.rectangle(mask, (left, top), (right, bottom), (255, 255, 255), -1)

    # 使用 inpaint 修复被标记的文字区域
    # inpaint 方法可选 cv2.INPAINT_TELEA 或 cv2.INPAINT_NS
    inpainted = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    return inpainted


def translate_chinese_to_english(text_list):
    """
    使用 googletrans 将一组中文文本翻译为英文。
    返回与输入等长的英文翻译结果列表。
    """
    translator = Translator()
    results = []
    for txt in text_list:
        # 避免空字符串翻译报错
        if not txt.strip():
            results.append('')
            continue
        # 将中文翻译为英文
        res = translator.translate(txt, src='zh-cn', dest='en')
        results.append(res.text)
    return results


def paste_text_on_image(opencv_image, boxes, new_texts, font_path=FONT_PATH, font_size=40):
    """
    在去除文字后的图像上，按照对应位置，绘制新的英文文字。
    这里使用 PIL.ImageDraw 在 BGR -> RGB 的格式转换后操作，再转换回去。
    
    注意：若要模拟原先的艺术风格，需要自行调整字体、颜色、描边、阴影等。
         这里做一个最基础的文字叠加示范。
    """
    # OpenCV 的图像是 BGR 格式，需要先转为 RGB 再给 Pillow 用
    pil_image = Image.fromarray(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    # 加载字体
    font = ImageFont.truetype(font_path, font_size)

    for box, new_text in zip(boxes, new_texts):
        if not new_text.strip():
            continue

        # 文字区域信息
        left = box['left']
        top = box['top']
        width = box['width']
        height = box['height']

        # 这里简单做一个：文字放在 box 中心
        # 如果需要更加仿原风格，可调文字大小、颜色、倾斜度等等
        text_w, text_h = draw.textsize(new_text, font=font)

        # 计算文字绘制左上角坐标，使其在 box 中居中
        draw_x = left + (width - text_w) / 2
        draw_y = top + (height - text_h) / 2

        # 绘制文字：可以调整 fill=(R,G,B)、stroke_width、stroke_fill 等来模拟效果
        draw.text((draw_x, draw_y), new_text, font=font, fill=(255, 255, 255))

    # 绘制完成后，再把 RGB 转回 BGR
    result_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return result_img


def main():
    # 1) OCR 获取图片中的中文文字及其位置
    boxes = ocr_and_get_boxes(INPUT_IMAGE_PATH)
    if not boxes:
        print("未检测到可用中文文字或置信度太低，程序结束。")
        return

    # 2) 根据文字框坐标在原图上进行 inpainting 修复背景
    inpainted_img = remove_text_by_inpainting(INPUT_IMAGE_PATH, boxes)

    # 3) 将所有中文文本翻译成英文
    original_texts = [b['text'] for b in boxes]
    translated_texts = translate_chinese_to_english(original_texts)
    
    # （可选）在终端打印看一下对应关系
    for cn, en in zip(original_texts, translated_texts):
        print(f"{cn} -> {en}")

    # 4) 在已经去除文字的图像上叠加新的英文文本
    result_img = paste_text_on_image(inpainted_img, boxes, translated_texts, FONT_PATH, font_size=40)

    # 5) 保存结果图像
    cv2.imwrite(OUTPUT_IMAGE_PATH, result_img)
    print(f"结果已保存到: {OUTPUT_IMAGE_PATH}")


if __name__ == '__main__':
    main()
