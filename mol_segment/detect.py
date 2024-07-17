import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
#  by Shaoxuan Yuan at 2024/04/22
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt

# 从unet模块导入U_Net类
from mol_segment.unet import U_Net

# # 将U_Net模块的路径添加到系统路径中
# sys.path.append('D:/桌面/project/atomai')


def calculate_coverage(predicted_mask):
    """
    计算并返回掩码与原图面积的比值，即覆盖率。
    """
    if predicted_mask.ndim != 2:
        # change the shape of the mask to 2D
        predicted_mask = predicted_mask.squeeze()
    mask_area = np.sum(predicted_mask)
    # print(predicted_mask.shape)  # 应该输出类似 (304, 304)
    
    total_area = 304*304
    coverage = mask_area / total_area
    return coverage

def Segmented_image(img, output_folder, model_path='./mol_segment/unet_modelsingle.pth'):
    model = U_Net(in_ch=3, out_ch=1)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # 检查文件扩展名，处理支持的图片格式
    try:
        img = Image.fromarray(img).convert('RGB')
    except:
        pass
    img = img.resize((304, 304))
    img_array = np.array(img) / 255.0
    img_tensor = torch.from_numpy(img_array.transpose((2, 0, 1))).float().unsqueeze(0)

    with torch.no_grad():
        prediction = model(img_tensor)
        predicted_mask = torch.sigmoid(prediction).squeeze(0).cpu().numpy()

    thresholded_mask = (predicted_mask > 0.5).astype(np.float32)
    distance_map = distance_transform_edt(1 - thresholded_mask)
    distance_map_corrected = distance_map.squeeze()
    coverage = calculate_coverage(thresholded_mask)
    
    # 计算nemo_point并添加边缘约束
    margin = img.width // 10  # 图像宽度的十分之一作为边缘区域
    valid_region = distance_map_corrected[margin:-margin, margin:-margin]
    nemo_point = np.unravel_index(np.argmax(valid_region), valid_region.shape)
    nemo_point = (nemo_point[0] + margin, nemo_point[1] + margin)
    


    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 根据输入图片文件名生成输出路径
    # get the time now as the format in Hour-Minute-Second
    output_path = os.path.join(output_folder, f"coverage_{coverage}_Result_{time.strftime('%H-%M-%S',time.localtime(time.time()))}.png")
    # output_path = os.path.join(output_folder, f"coverage_{coverage}_Result_{os.path.basename(image_path)}")

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(img_array)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(thresholded_mask.squeeze(), cmap='gray')
    ax[1].set_title('Thresholded Mask')
    ax[1].axis('off')

    ax[2].imshow(img_array)
    ax[2].imshow(distance_map_corrected, alpha=0.5, cmap='jet')
    ax[2].scatter(*nemo_point[::-1], color='red')  # 标记调整后的Nemo点, here the nome is a point
    ax[2].set_title('Distance Map')
    ax[2].axis('off')

    fig.savefig(output_path)
    plt.close(fig)
    # print(nemo_point )
    # exchange the nemo_point X and Y
    nemo_point = (nemo_point[1], nemo_point[0])

    nemo_point = (((nemo_point[0])-304*0.5)/304, ((nemo_point[1])-304*0.5)/304) # the nemo_point is normalized and X Y is exchanged and change the fomat from (X,Y) to matrix so the Y → -Y
    return nemo_point, coverage

if __name__ == '__main__':

    # 示例使用
    image_path = './mol_segment/detect/image1/002 [Z_fwd] image.png'
    output_folder = './mol_segment/results1'
    model_path = './mol_segment/unet_modelsingle.pth'
    img = Image.open(image_path)
    Segmented_image(img, output_folder,model_path)

