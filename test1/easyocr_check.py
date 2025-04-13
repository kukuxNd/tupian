import easyocr
import torch
print(torch.__version__)
# 初始化EasyOCR阅读器
reader = easyocr.Reader(['ch_sim'])

# 定义一个函数来检查图片中文字
def check_text_in_image(image_path):
    # 使用EasyOCR读取图片中的文字
    result = reader.readtext(image_path)
    
    # 输出识别到的文字
    for detection in result:
        print(detection[1])  # detection[1] 是识别到的文字

# 示例用法
image_path = 'o_png/8S.png'  # 替换为你的图片路径
check_text_in_image(image_path) 