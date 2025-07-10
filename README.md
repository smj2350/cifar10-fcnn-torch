### 라이브러리
- PyTorch, matplotlib

### 모델 구조
- 입력 크기 : 1 × 28 × 28
- 가중치 초기화 : He(Kaiming) Normal
- Bias : 전부 0으로 초기화 (기울기 소실 및 폭주 완화)
- 활성 함수 : ReLU, LogSoftmax + NLLLoss

레이어:
1. Conv2d(1→16, 3×3, padding=1) → ReLU → MaxPool(2)
2. Conv2d(16→32, 3×3, padding=1) → ReLU → MaxPool(2)
3. Linear(32·7·7 → 128) → ReLU
4. Linear(128 → 10) (Logits)

### 하이퍼파라미터
- Optimizer: Adam
- Learning Rate: 0.001
- Epochs: 20
- Loss: CCEE

### 파이프라인
- Transform : ToTensor() → Normalize(μ = 0.5, σ = 0.5)
- 배치 크기 : 64
- DataLoader : 매 epoch 마다 랜덤 셔플(IID)

### Evaluate
- model.eval() -> torch.no_grad()로 gradient 미계산
- 평균 손실 = (미니배치 손실 합) / (len(loader))
- 정확도 = (TP 개수) / 전체 샘플
