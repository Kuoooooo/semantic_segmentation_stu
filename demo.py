"""
__author__      = 'kwok'
__time__        = '2021/11/8 14:12'
"""

import torch
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms, models
import numpy as np


def decode_seg(image, source, nc=21):
    label_colors = np.array([
        (0, 0, 0),  # 0=background
        # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
        (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
        # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
        (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
        # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
        (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
        # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
        (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)
    ])
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    # 每个像素对应的类别赋予相应的颜色
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    # 这个就是语义分割的彩色图
    rgb = np.stack([r, g, b], axis=2)
    plt.imshow(rgb)
    # plt.axis('off')
    plt.show()

    foreground = cv2.imread(source)

    foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
    foreground = cv2.resize(foreground, (r.shape[1], r.shape[0]))

    # 这里使用一张全白的图像作为替换的背景
    background = 255 * np.ones_like(rgb).astype(np.uint8)

    foreground = foreground.astype(float)
    background = background.astype(float)

    # 背景的值为0，这里以0为阈值由分割图得到mask，分离出背景
    # 在二值化之前需要先将分割图转化成灰度图，否则thresh分别作用于每个通道
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    th, binary = cv2.threshold(np.array(gray), 0, 255, cv2.THRESH_BINARY)
    # 由于边缘很锐利，所以做一个模糊平滑边缘，这样在过渡的地方看起来自然一些
    mask = cv2.GaussianBlur(binary, (7, 7), 0)
    # 将mask转换成3通道
    alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    alpha = alpha.astype(float) / 255
    plt.imshow(alpha)
    # plt.axis('off')
    plt.show()

    # alpha混合
    foreground = cv2.multiply(alpha, foreground)

    background = cv2.multiply(1.0 - alpha, background)

    outImage = cv2.add(foreground, background)

    return outImage / 255


if __name__ == '__main__':
    # 判断 GPU 是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 模型加载
    model = models.segmentation.deeplabv3_resnet101(pretrained=True).to(device)
    # 本地模型加载，strict 选 False
    # model = models.segmentation.deeplabv3_resnet101().to(device)
    # model.load_state_dict(torch.load('weights/deeplabv3_resnet101_coco-586e9e4e.pth', map_location=device), strict=False)
    model.eval()
    # 加载图片
    image_path = 'huge.jpg'
    input_image = cv2.imread(image_path)
    # 图片转换
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    input_batch = input_batch.to(device)

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    rgb = decode_seg(output_predictions, image_path)

    plt.imshow(rgb)
    plt.show()
