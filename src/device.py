import torch

# Verifica quante GPU sono disponibili
print(torch.cuda.device_count())

# Elenca i nomi delle GPU
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
