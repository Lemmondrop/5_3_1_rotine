import torch

print('CUDA Available:', torch.cuda.is_available())
print('Divice Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')