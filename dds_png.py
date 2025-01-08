import os
import imageio
from PIL import Image

def dds_to_png(dds_path, png_path):
    # 使用imageio读取DDS文件
    try:
        image = imageio.imread(dds_path, format='DDS')
        # 将图像数据转换为Pillow图像对象
        pil_image = Image.fromarray(image)
        # 保存为PNG文件
        pil_image.save(png_path)
        print(f"转换成功: {dds_path} -> {png_path}")
    except Exception as e:
        print(f"转换失败: {dds_path} - {str(e)}")

def batch_convert_dds_to_png(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有DDS文件
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.dds'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.png')
            dds_to_png(input_path, output_path)

# 使用示例
input_folder = "t_dds"  # 输入文件夹路径，包含DDS文件
output_folder = "o_png"  # 输出文件夹路径，保存PNG文件
batch_convert_dds_to_png(input_folder, output_folder)