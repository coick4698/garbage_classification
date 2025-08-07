import torch
import cv2
import numpy as np
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def preprocess_image(img_path, image_size=224):
    img = cv2.imread(img_path, 1)[:, :, ::-1]  # BGR -> RGB
    img = cv2.resize(img, (image_size, image_size))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]    # ImageNet std
        )
    ])
    input_tensor = transform(img).unsqueeze(0)  # [1, 3, H, W]

    return img, input_tensor

def run_gradcam(model, target_layer, img_path, target_category=None, use_cuda=True):

    model.eval()
    if use_cuda:
        model = model.cuda()

    raw_img, input_tensor = preprocess_image(img_path)
    if use_cuda:
        input_tensor = input_tensor.cuda()

    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=use_cuda)

    grayscale_cam = cam(input_tensor=input_tensor, targets=None if target_category is None else [target_category])
    grayscale_cam = grayscale_cam[0, :]

    visualization = show_cam_on_image(raw_img / 255.0, grayscale_cam, use_rgb=True)

    return visualization
