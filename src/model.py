# 모델 정의
import torch.nn as nn
from torchvision import models

# MobilenetV2
def get_mobilenetv2(num_classes=12):
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel,num_classes)
    return model

# Densenet121
def get_densenet121(num_classes=12):
    model = models.densenet121(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features,num_classes)
    return model

# Squeezenet
def get_squeezenet(num_classes=12):
    model = models.squeezenet1_0(pretrained=True)
    model.classifier[1] = nn.Conv2d(512,num_classes, kernel_size=1)
    return model

# ShufflenetV2
def get_shufflenetv2(num_classes=12):
    model = models.shufflenet_v2_x1_0(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features,num_classes)
    return model
