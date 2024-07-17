
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# 添加 unet.py 所在的目录到系统路径
sys.path.append('D:/桌面/project/atomai')
from unet import U_Net  # 从 unet.py 中导入 U_Net 类

# 加载模型
model = U_Net(in_ch=3, out_ch=1)
#model = UNet(n_channels=3, out_channels = 1)
model.load_state_dict(torch.load('D:/桌面/project/atomai/unet_model.pth', map_location=torch.device('cpu')))
model.eval()

# 选择并处理一个图像进行预测
image_path = 'D:/桌面/project/atomai/detect/image/003 [Z_fwd] image.png'
img = Image.open(image_path)
img = img.resize((304, 304))  # 根据模型要求调整图像大小
img_array = np.array(img) / 255.0
img_tensor = torch.from_numpy(img_array.transpose((2, 0, 1))).float().unsqueeze(0)  # 转换为 NCHW 格式

# 进行预测
with torch.no_grad():
    prediction = model(img_tensor)
    predicted_mask = torch.sigmoid(prediction).squeeze(0).cpu().numpy()

# 阈值化以可视化前景
thresholded_mask = (predicted_mask > 0.5).astype(np.float32)

# 可视化结果
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(img_array)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_array)
plt.imshow(np.squeeze(thresholded_mask), alpha=0.5, cmap='jet')  # 使用 np.squeeze 去掉单维度
plt.title('Predicted Mask')
plt.axis('off')

plt.show()
