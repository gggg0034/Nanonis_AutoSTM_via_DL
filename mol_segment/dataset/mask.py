# import json
# import numpy as np
# import cv2
# from matplotlib import pyplot as plt

# # 假设我们处理一个JSON文件
# json_file = 'D:/桌面/project/atomai/testdata/1.json'  # 替换为实际的JSON文件路径
# with open(json_file) as f:
#     data = json.load(f)

# # 创建一个与图像尺寸相同的空数组
# image_height = data['imageHeight']
# image_width = data['imageWidth']
# mask = np.zeros((image_height, image_width), dtype=np.uint8)

# # 对于JSON文件中的每个标注多边形
# for shape in data['shapes']:
#     points = np.array(shape['points'], dtype=np.int32)
#     cv2.fillPoly(mask, [points], color=(255))  # 255用于二元分割

# # 保存掩码图像
# mask_file = 'path_to_save_mask.png'  # 替换为掩码保存路径
# cv2.imwrite(mask_file, mask)

# # 可视化掩码
# plt.imshow(mask, cmap='gray')
# plt.show()

#  by Shaoxuan Yuan at 2024/04/22
import json
import os

import cv2
import numpy as np

# from matplotlib import pyplot as plt

json_dir = './dataset/image'
mask_dir = './dataset/mask'
if not os.path.exists(mask_dir):
    os.makedirs(mask_dir)

print("开始处理文件...")
for file in os.listdir(json_dir):
    if file.endswith(".json"):
        try:
            json_file = os.path.join(json_dir, file)
            with open(json_file) as f:
                data = json.load(f)

            image_height = data['imageHeight']
            image_width = data['imageWidth']
            mask = np.zeros((image_height, image_width), dtype=np.uint8)

            for shape in data['shapes']:
                points = np.array(shape['points'], dtype=np.int32)
                cv2.fillPoly(mask, [points], color=(255))

            mask_file = os.path.join(mask_dir, os.path.splitext(file)[0] + '.png')
            cv2.imwrite(mask_file, mask)
            print(f"已保存掩码: {mask_file}")

        except Exception as e:
            print(f"处理文件 {file} 时发生错误: {e}")

# 可视化部分代码已注释，因为在大量数据处理时可能不需要
