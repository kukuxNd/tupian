import cv2
import pytesseract

# 读取图像
image = cv2.imread('o_png/8S.png')

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 二值化处理
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 使用 Tesseract OCR 进行文字检测
custom_config = r'--oem 3 --psm 6'
details = pytesseract.image_to_data(binary, output_type=pytesseract.Output.DICT, config=custom_config, lang='chi_sim')

# 遍历检测到的文字
n_boxes = len(details['text'])
for i in range(n_boxes):
    if int(details['conf'][i]) > 60:  # 置信度阈值
        (x, y, w, h) = (details['left'][i], details['top'][i], details['width'][i], details['height'][i])
        # 在原图上绘制矩形框
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 显示结果
cv2.imshow('Image with Chinese Characters', binary)
cv2.waitKey(0)
cv2.destroyAllWindows()