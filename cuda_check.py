import torch

if torch.cuda.is_available():
    print("GPU is accessible.")
else:
    print("GPU is not accessible.")
