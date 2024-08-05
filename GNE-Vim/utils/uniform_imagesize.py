import os
import cv2
from torchvision import transforms

from utils.process_image import RandCrop

crop_size = 224
num_crop = 1

transform = transforms.Compose(
    [
        RandCrop(crop_size, num_crop),
    ]
)

imagePaths = [r'D:\hqy\datasets\IQA\CLIVE\distorted_images',
              r'D:\hqy\datasets\IQA\KonIQ\distorted_images',
              'D:\hqy\datasets\IQA\LIVE2005\distorted_images']

savePaths = [r'D:\hqy\datasets\IQA\CLIVE\distorted_images\distorted_images_unified',
             r'D:\hqy\datasets\IQA\KonIQ\distorted_images_unified',
             r'D:\hqy\datasets\IQA\LIVE2005\distorted_images_unified']

for i, imagePath in enumerate(imagePaths):
    imageNames = os.listdir(imagePath)
    savePath = savePaths[i]

    for imageName in imageNames:
        r_img = cv2.imread(os.path.join(imagePath, imageName), cv2.IMREAD_COLOR)
        r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)
        r_img = transform(r_img)
        print(imageName)