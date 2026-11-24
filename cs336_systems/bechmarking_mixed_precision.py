import torch
from torch import nn as nn 

class ToyModel(nn.Module):
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()
    def forward(self, x):
        fc1 = self.fc1(x)
        print(f"fc1 out {fc1.dtype}")
        relu = self.relu(fc1)
        print(f"relu out {relu.dtype}")
        out = self.ln(relu)
        print(f"out {out.dtype}")
        final = self.fc2(out)
        return final
batch_size = 2
in_features = 8
out_features = 4 

module = ToyModel(in_features=in_features, out_features=out_features)
module.to(device="cuda")
input = torch.randn(batch_size, in_features, device="cuda", dtype=torch.float32)

with torch.autocast(device_type="cuda", dtype=torch.float16):
    fp = module(input)
    print("fc1 weights: ", module.fc1.weight.dtype)

    target = torch.randint(0, out_features, (batch_size,), device="cuda")
    loss = nn.functional.cross_entropy(fp, target)
    print(f"loss {loss.dtype}")
loss.backward()
print("input dtype ", input.dtype)
print("fp dtype", fp.dtype)
print("grad dtype : ", module.fc1.weight.grad.dtype)
