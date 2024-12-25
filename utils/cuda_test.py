import torch

# 检查是否有可用的 GPU
if torch.cuda.is_available():
    print("Training on GPU.")
else:
    print("Training on CPU.")

