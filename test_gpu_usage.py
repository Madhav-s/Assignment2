import torch
import time

print("Testing GPU usage...")

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create a simple model and data
model = torch.nn.Linear(1000, 1000).to(device)
data = torch.randn(100, 1000).to(device)

# Time a forward pass
start = time.time()
for _ in range(100):
    output = model(data)
end = time.time()

print(".2f")

# Check GPU memory
if torch.cuda.is_available():
    print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved()/1024**2:.1f} MB")