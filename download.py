import timm
import torch

model = timm.create_model("convnextv2_tiny", pretrained=False)

x = torch.randn(1, 3, 256, 256)

x = model.stem(x)
print("stem:", x.shape)

for i, stage in enumerate(model.stages):
    x = stage(x)
    print(f"stage{i}:", x.shape)