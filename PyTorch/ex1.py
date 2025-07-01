import torch

x = torch.tensor([2.0], requires_grad=True)
y = x**2 + 3 * x
y.backward()

print("x:", x.item())
print("y:", y.item())
print("dy/dx:", x.grad.item())
