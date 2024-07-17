#  by Shaoxuan Yuan at 2024/04/22
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 图像和掩码的文件夹路径
image_folder = './mol_segment/dataset/augmented_images1'
mask_folder = './mol_segment/dataset/augmented_masks'

# 加载图像和掩码
images = []
masks = []

desired_size = (304, 304)  # 您希望的统一图像尺寸

for filename in os.listdir(image_folder):
    if filename.endswith('.png'):
        img_path = os.path.join(image_folder, filename)
        # 修改掩码文件名以匹配原图文件名
        mask_filename = filename.replace('aug_', 'aug_mask_')
        mask_path = os.path.join(mask_folder, mask_filename)

        # 加载图像和掩码
        img = Image.open(img_path)
        mask = Image.open(mask_path)

        # 调整图像和掩码尺寸
        img = img.resize(desired_size)
        mask = mask.resize(desired_size)

        # 转换为 NumPy 数组并归一化
        img_array = np.asarray(img) / 255.0
        mask_array = np.asarray(mask)

        # 将所有非零值的掩码转换为 1，零值保持为 0
        mask_array = np.where(mask_array != 0, 1, 0).astype(np.int32)

        images.append(img_array)
        masks.append(mask_array)

# 现在转换为 NumPy 数组
images = np.array(images, dtype=np.float32)
masks = np.array(masks, dtype=np.int32)

# 划分训练和测试集
images_train, images_test, masks_train, masks_test = train_test_split(images, masks, test_size=0.2)

# 保存数据
np.save('./mol_segment/dataset/train_images1.npy', images_train)
np.save('./mol_segment/dataset/train_masks1.npy', masks_train)
np.save('./mol_segment/dataset/test_images1.npy', images_test)
np.save('./mol_segment/dataset/test_masks1.npy', masks_test)



# 创建 PyTorch 数据集
class CustomDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        if self.transform:
            image = self.transform(image)

        return image, mask

# 定义变换（如果需要）
transform = transforms.Compose([
    transforms.ToTensor(),
    # 添加其他任何您需要的变换
])

# 创建 PyTorch 数据集
train_dataset = CustomDataset(images_train, masks_train, transform=transform)
test_dataset = CustomDataset(images_test, masks_test, transform=transform)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# 现在您可以在训练循环中使用 train_loader 和 test_loader
