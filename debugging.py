import torch
print(f"torch.__version__:{torch.__version__}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "No GPU found")
print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")