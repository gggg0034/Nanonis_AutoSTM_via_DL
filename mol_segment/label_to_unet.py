#  by Shaoxuan Yuan at 2024/04/22
from PIL import Image, ImageDraw
import json

def fill_polygon(json_file_path, output_file_path):
    # 读取JSON文件中的标注信息
    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)

    # 获取图片的宽度和高度
    width = json_data['imageWidth']
    height = json_data['imageHeight']

    # 创建白色背景图像
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)

    # 获取polygon标注信息
    for annotation in json_data['shapes']:
        # 获取多边形的点坐标并将浮点数转换为整数
        points = [(int(x), int(y)) for x, y in annotation['points']]

        # 将polygon区域填充为白色
        draw.polygon(points, fill='white')

    # 剩余区域填充为黑色
    draw.rectangle([0, 0, width, height], fill='black')

    # 保存输出图像
    img.save(output_file_path)


# 用法示例
if __name__ == "__main__":
    json_file_path = "dataset/labels/007 [Z_fwd] graph.json"  # 替换为你的JSON文件路径
    output_file_path = "dataset/labels/output_image.png"  # 替换为输出图像的路径
    fill_polygon(json_file_path, output_file_path)
