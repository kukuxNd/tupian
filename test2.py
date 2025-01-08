import cv2
import pytesseract
import numpy as np
import os

def remove_chinese_characters(image_path, output_path):
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片: {image_path}")
        return

    # 将图片转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用pytesseract进行文字识别
    data = pytesseract.image_to_data(gray, lang='chi_sim', output_type=pytesseract.Output.DICT)

    # 输出识别到的文字内容
    recognized_text = ' '.join([text for text in data['text'] if text.strip()])
    print(f"识别到的文字内容: {recognized_text}")

    # 创建一个与原图大小相同的全黑掩码
    mask = np.zeros_like(gray)

    # 遍历识别到的文字区域
    for i in range(len(data['text'])):
        if data['text'][i].strip():  # 只处理非空文本
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            # 在掩码上绘制白色矩形，覆盖中文字符
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    # 使用图像修复算法恢复背景
    restored_image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

    # 保存结果
    cv2.imwrite(output_path, restored_image)
    print(f"处理完成: {output_path}")

def batch_process_images(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有图片
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            remove_chinese_characters(input_path, output_path)

# 使用示例
input_folder = "t_img"  # 输入文件夹路径
output_folder = "out_img"  # 输出文件夹路径
batch_process_images(input_folder, output_folder)