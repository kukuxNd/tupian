import cv2
import pytesseract

def read_chinese_characters(image_path):
    # 检查图片是否为dds格式
    if image_path.lower().endswith('.dds'):
        # 读取dds格式图片
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    else:
        # 读取非dds格式图片
        img = cv2.imread(image_path)
    
    # 检查图片是否读取成功
    if img is None:
        print(f"无法读取图片文件：{image_path}")
        return None
    
    # 将图片转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 使用pytesseract进行文字识别
    text = pytesseract.image_to_string(gray, lang='chi_sim')
    return text

# 调用函数
image_path = "./img/p3.png"
result = read_chinese_characters(image_path)
if result is not None:
    print("识别到的文字：", result)
else:
    print("图片读取失败，无法识别文字。")
