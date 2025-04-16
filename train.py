# 모델 저장하기
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 2. 데이터 전처리
# 데이터 정규화
symbol = "SOLUSDT"
df = pd.read_csv(f"dataset/ohlcv/{symbol}.csv")
df = df.drop("time", axis=1)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)


# 3. 시퀀스 데이터 생성 함수 (window size = 100)
def create_sequences(data, seq_length=100, pred_length=10):
    X = []
    y = []
    for i in range(len(data) - seq_length - pred_length + 1):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length:i+seq_length+pred_length])
    return np.array(X), np.array(y)


# 시퀀스 데이터 생성
seq_length = 100
pred_length = 10
X, y = create_sequences(scaled_data, seq_length, pred_length)

# 데이터 형태 확인
print(f"X shape: {X.shape}")  # (900, 100, 4) - (샘플 수, 시퀀스 길이, 특성 수)
print(f"y shape: {y.shape}")  # (900, 4) - (샘플 수, 특성 수)

# 4. 데이터 분할 (훈련/검증/테스트)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=False)

# 5. PyTorch용 텐서로 변환
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.FloatTensor(y_val)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

# 6. 데이터로더 생성
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# 7. LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, pred_length):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_length = pred_length

        # LSTM 레이어와 드롭아웃 정의
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        # 출력 레이어 정의
        self.fc = nn.Linear(hidden_size, output_size * pred_length)

    def forward(self, x):
        # 초기 은닉 상태 및 셀 상태 초기화
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM 순전파 (출력, (마지막 은닉 상태, 마지막 셀 상태))
        out, _ = self.lstm(x, (h0, c0))

        # 마지막 시점의 출력만 사용
        out = self.fc(out[:, -1, :])

        # 출력을 (배치, 예측길이, 특성 수) 형태로 변환
        out = out.view(x.size(0), self.pred_length, -1)
        return out


# 8. 모델 초기화 및 학습 설정
input_size = 5      # 특성의 수
hidden_size = 64    # LSTM 은닉 유닛의 수
num_layers = 2      # LSTM 레이어의 수
output_size = 5     # 예측할 특성의 수
pred_length = 10    # 예측할 시점의 수

# 모델, 손실 함수, 옵티마이저 초기화
model = LSTMModel(input_size, hidden_size, num_layers, output_size, pred_length)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 9. 모델 학습 함수
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # 훈련 모드
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            # 그래디언트 초기화
            optimizer.zero_grad()

            # 순전파
            outputs = model(inputs)

            # 손실 계산
            loss = criterion(outputs, targets)

            # 역전파 및 옵티마이저 스텝
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)

        # 검증 모드
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    return train_losses, val_losses


# 10. 모델 학습
num_epochs = 50
train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)



# 모델 저장하기; 타임스탬프와 검증 손실을 파일명에 포함
timestamp = time.strftime("%Y%m%d-%H%M%S")
val_loss = val_losses[-1]
model_filename = f'lstm_model_{timestamp}_valloss_{val_loss:.4f}.pth'

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_losses': train_losses,
    'val_losses': val_losses,
    'scaler': scaler,
}, model_filename)

print(f"모델이 '{model_filename}' 파일로 저장되었습니다.")

# 11. 손실 시각화
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
# plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss')
# plt.legend()
# plt.grid(True)
# plt.show()

# 12. 모델 평가
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor)
    test_loss = criterion(test_predictions, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')


# 모델 불러오기
def load_model(model_path, input_size, hidden_size, num_layers, output_size, pred_length=10):
    # 모델 구조 초기화
    model = LSTMModel(input_size, hidden_size, num_layers, output_size, pred_length)

    # 저장된 모델 파일 불러오기
    checkpoint = torch.load(model_path)

    # 모델 가중치 불러오기
    model.load_state_dict(checkpoint['model_state_dict'])

    # 필요한 경우 옵티마이저 불러오기
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 손실 기록 불러오기 (선택 사항)
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']

    # 스케일러 불러오기
    scaler = checkpoint['scaler']

    return model, optimizer, train_losses, val_losses, scaler


# 모델 불러오기 사용 예시
loaded_model, loaded_optimizer, loaded_train_losses, loaded_val_losses, loaded_scaler = load_model(
    'lstm_model.pth',
    input_size=4,
    hidden_size=64,
    num_layers=2,
    output_size=4,
    pred_length=10
)

# 불러온 모델로 예측하기
loaded_model.eval()
with torch.no_grad():
    test_predictions = loaded_model(X_test_tensor)
    test_loss = criterion(test_predictions, y_test_tensor)
    print(f'불러온 모델의 테스트 손실: {test_loss.item():.4f}')

# 13. 예측 결과 시각화
# 스케일링 복원
test_predictions_np = test_predictions.numpy()
y_test_np = y_test_tensor.numpy()

# 실제 값과 예측 값의 비교 (첫 번째 특성에 대해서만)
plt.figure(figsize=(12, 6))
plt.plot(y_test_np[:, 0], label='Actual')
plt.plot(test_predictions_np[:, 0], label='Predicted')
plt.title('LSTM Model: Actual vs Predicted (Feature 1)')
plt.xlabel('Time Steps')
plt.ylabel('Normalized Value')
plt.legend()
plt.grid(True)
plt.show()


# 14. 새로운 데이터로 예측하는 함수
def predict_future(model, last_sequence, num_predictions, scaler):
    model.eval()
    current_sequence = last_sequence.clone()
    predictions = []

    with torch.no_grad():
        for _ in range(num_predictions):
            # 현재 시퀀스에 대한 예측 (10개 시점 예측)
            current_input = current_sequence.unsqueeze(0)  # 배치 차원 추가
            predicted_sequence = model(current_input)  # (1, 10, 4) 형태

            # 첫 번째 예측만 저장
            first_pred = predicted_sequence[0, 0].numpy()
            predictions.append(first_pred)

            # 시퀀스 업데이트 (가장 오래된 시점 제거, 예측 값 추가)
            current_sequence = torch.cat([current_sequence[1:], predicted_sequence[0, 0].unsqueeze(0)], dim=0)

    return np.array(predictions)


# 15. 미래 예측 수행
# 테스트 세트의 마지막 시퀀스 가져오기
last_sequence = X_test_tensor[-1]
num_future_predictions = 50

future_predictions = predict_future(model, last_sequence, num_future_predictions, scaler)

# 16. 미래 예측 시각화
plt.figure(figsize=(12, 6))
plt.plot(range(len(y_test_np)), y_test_np[:, 0], label='Historical Data')
plt.plot(range(len(y_test_np), len(y_test_np) + num_future_predictions), future_predictions[:, 0],
         label='Future Predictions', linestyle='--')
plt.title('LSTM Model: Historical Data and Future Predictions (Feature 1)')
plt.xlabel('Time Steps')
plt.ylabel('Normalized Value')
plt.legend()
plt.grid(True)
plt.show()