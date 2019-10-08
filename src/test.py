import torch
from torchvision.models import resnet18, alexnet
from ai import ArithmeticIntensity
model = resnet18()
print(model)
ai = ArithmeticIntensity(model=model, input_dims=(1, 3, 224, 224))
ai.get_metrics()