#  by Shaoxuan Yuan at 2024/04/22
from PIL import Image
import numpy as np

def image_to_numpy(image_path):
    # 使用PIL库打开图片
    image = Image.open(image_path)
    # 将图片转换为NumPy数组
    image_array = np.array(image)
    return image_array

def save_numpy_as_npy(numpy_array, save_path):
    # 保存为.npy格式
    np.save(save_path, numpy_array)

if __name__ == "__main__":
    # 图片文件路径
    input_image_path = "atomai/dataset/image/1_json/label.png"
    # 保存的.npy文件路径
    npy_save_path = "num/image"

    # 将图片转换为NumPy数组
    numpy_array = image_to_numpy(input_image_path)

    # 保存NumPy数组为.npy文件
    save_numpy_as_npy(numpy_array, npy_save_path)

    print("图片已成功转换为NumPy数组并保存为.npy文件。")


