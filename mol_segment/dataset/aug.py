
#  by Shaoxuan Yuan at 2024/04/22
import cv2
import os
from albumentations import HorizontalFlip, Rotate, Compose, ElasticTransform, CoarseDropout

# 数据增强的变换序列
def get_transforms():
    return Compose([
        HorizontalFlip(p=0.5),
        Rotate(limit=45, p=0.5),
        ElasticTransform(p=0.5),
        CoarseDropout(p=0.5)
    ])

def augment_images(image_dir, mask_dir, save_image_dir, save_mask_dir, augmentations_per_image):
    # 检查并创建目录
    if not os.path.exists(save_image_dir):
        os.makedirs(save_image_dir)
    if not os.path.exists(save_mask_dir):
        os.makedirs(save_mask_dir)

    transform = get_transforms()

    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        mask_path = os.path.join(mask_dir, image_name)

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        for i in range(augmentations_per_image):
            augmented = transform(image=image, mask=mask)
            augmented_image = augmented['image']
            augmented_mask = augmented['mask']

            aug_image_name = f"aug_{i}_{image_name}"
            aug_mask_name = f"aug_mask_{i}_{image_name}"

            cv2.imwrite(os.path.join(save_image_dir, aug_image_name), augmented_image)
            cv2.imwrite(os.path.join(save_mask_dir, aug_mask_name), augmented_mask)

# 使用函数
image_directory = 'dataset/image'  # 原图像目录
mask_directory = 'dataset/mask'    # 掩码目录
save_image_directory = 'dataset/augmented_images'  # 增强图像保存目录
save_mask_directory = 'dataset/augmented_masks'    # 增强掩码保存目录
augmentations_per_image = 15  # 每张图片生成的增强版本数量

augment_images(image_directory, mask_directory, save_image_directory, save_mask_directory, augmentations_per_image)
