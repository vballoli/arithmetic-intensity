import torch
from torchvision.models import alexnet, vgg11, mobilenet, mnasnet0_5, resnet18
from ai import ArithmeticIntensity
from utils import *
from thop import profile

model = resnet18()
model.eval()
model.apply(add_hook)
with torch.no_grad():
    model(torch.ones(1, 3, 224, 224))
total_ai = 0
for m in model.modules():
    if len(list(m.children())) > 0:  # skip for non-leaf module
        continue
    total_ai += m.total_ai
print(total_ai.item())

# for c in model.children():
#     mo
#ai = ArithmeticIntensity(model=model, input_dims=(1, 3, 224, 224))
#ai.get_metrics()