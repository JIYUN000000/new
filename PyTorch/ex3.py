import torch
import torch.nn as nn
import torch.optim as optim

# 학습 데이터 (x와 y의 관계: y = 2x + 1)
x_train = torch.tensor([[1.0],[2.0],[3.0],[4.0]])
y_train = torch.tensor([[3.0],[5.0],[7.0],[9.0]])

# 2. 모델 정의
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)

    def forward(self,t):
        return self.linear(t)

model = LinearModel()

# 손실 함수와 옵티마이저 설정
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)

# 학습 전 가중치 확인
print("Before training:")
print("Weight: ", model.linear.weight.data)
print("Bias: ", model.linear.bias.data)

# 학습 루프
for epoch in range(100):
    output = model(x_train) # 예측값
    loss = criterion(output,y_train) # 손실 계산

    optimizer.zero_grad() # 기울기 초기화
    loss.backward() # 역전파 (오차를 기반으로 역전파 수행)
    optimizer.step() # 가중치 update

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# 학습 후 가중치 확인
print("After training:")
print("Weight: ", model.linear.weight.data)
print("Bias: ", model.linear.bias.data)

# 새 입력값에 대해 예측해보기
new_x= torch.tensor([[5.0]])
pred = model(new_x)
print("x=5.0일 때 예측값:", pred.item())