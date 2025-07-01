# nn.Module로 간단한 모델 만들기
# PyTorch에서는 nn.Linear 레이어가 내부적으로 weight와 bias를 자동으로 만들어주고 학습대상으로 관리함
import torch
import torch.nn as nn

# 간단한 선형 모델: y = Wx + b

class LinearModel(nn.Module):
    # 신경망 설계 공간
    def __init__(self):
        super().__init__()
        # 신경망 구조를 내 맘대로 정의 (선형 계층 1개)
        self.linear = nn.Linear(1,1)

    def forward(self, t):
        # 모델에 t를 넣었을 때 어떻게 계산할지 정의
        return self.linear(t)
        # 선형 계층을 통해 출력값 계산

# 모델 인스턴스 생성
model = LinearModel()

# 예시 입력
x = torch.tensor([[2.0]])
output = model(x)
print("모델 출력: ", output)
print("weight:", model.linear.weight)
print("bias:", model.linear.bias)