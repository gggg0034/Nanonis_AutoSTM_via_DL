#  by Shaoxuan Yuan at 2024/04/22
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from unet import U_Net  # 从您的 UNet 文件导入模型类
# from old_unet_model import UNet  # 从您的 UNet 文件导入模型类
# 加载数据集
train_images = np.load('./mol_segment/dataset/train_images1.npy')
train_masks = np.load('./mol_segment/dataset/train_masks1.npy')
val_images = np.load('./mol_segment/dataset/test_images1.npy')
val_masks = np.load('./mol_segment/dataset/test_masks1.npy')

# 将 NumPy 数组转换为 PyTorch Tensor
train_images = torch.Tensor(train_images.transpose((0, 3, 1, 2)))  # 转换为 NCHW 格式
train_masks = torch.Tensor(train_masks).unsqueeze(1)
val_images = torch.Tensor(val_images.transpose((0, 3, 1, 2)))
val_masks = torch.Tensor(val_masks).unsqueeze(1)

# 创建 TensorDataset 和 DataLoader
train_dataset = TensorDataset(train_images, train_masks)
val_dataset = TensorDataset(val_images, val_masks)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# 初始化模型、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = U_Net().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练过程
num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    # 打印训练损失
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

    # 可选：在验证集上验证模型
    # ...

# 保存模型权重
torch.save(model.state_dict(), 'unet_modelsingle.pth')
