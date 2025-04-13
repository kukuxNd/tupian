from rapidocr_onnxruntime import RapidOCR
from PIL import Image
import cv2

img_path = 'o_png/8S.png'
img = cv2.imread(img_path)
if img is None:
    print("图像加载失败，请检查文件路径或图像格式")
else:
    print("图像加载成功")

img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


engine = RapidOCR()
result, elapse = engine(img_pil)
print(result)
print(elapse)