import easyocr
import cv2

# 读取图像
image_path = 'o_png/8S.png'
image = cv2.imread(image_path)

# 创建 easyocr Reader 对象，指定使用中文
reader = easyocr.Reader(['ch_sim'])

# 识别图像中的文字
results = reader.readtext(image)

# 打印识别结果
for (bbox, text, prob) in results:
    print(f"识别出的文字: {text}, 置信度: {prob:.2f}")