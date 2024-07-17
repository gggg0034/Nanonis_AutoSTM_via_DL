#  by Shaoxuan Yuan at 2024/04/22
import os
from PIL import Image
import numpy as np

def convert_to_three_channel_gray(input_folder, output_folder):
    """
    将指定文件夹中的所有彩色图片转换为三通道灰白图片。
    
    参数:
        input_folder (str): 包含图片的文件夹路径。
        output_folder (str): 灰白图片保存的文件夹路径。
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 处理输入文件夹中的每张图片
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(file_path).convert('L')  # 转换图片为灰度图
            gray_array = np.array(img)
            three_channel_gray_array = np.stack((gray_array,)*3, axis=-1)  # 将灰度数组复制到三个通道
            three_channel_gray_image = Image.fromarray(three_channel_gray_array)
            three_channel_gray_image.save(os.path.join(output_folder, filename))  # 保存修改后的图片

# 示例使用
input_folder = 'detect/image'
output_folder = 'detect/image1'
convert_to_three_channel_gray(input_folder, output_folder)
