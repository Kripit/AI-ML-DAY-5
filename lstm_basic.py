import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ===================== CONFIGURATION =====================
seq_length = 20
input_size = 1
hidden_size = 30
output_size = 1
batch_size = 32
epochs = 60
lr = 0.01

# ===================== DATA GENERATION =====================
t = np.linspace(0, 100, 1000)
data = np.sin(0.1 * t) + np.random.normal(0, 0.1, 1000)
data = data.reshape(-1, 1)

# ===================== CREATE SEQUENCES =====================
X, y = [], []
for i in range(len(data) - seq_length):
    X.append(data[i:i + seq_length])
    y.append(data[i + seq_length])

X = np.array(X)
y = np.array(y)
X = torch.FloatTensor(X)
y = torch.FloatTensor(y)

dataset = torch.utils.data.TensorDataset(X, y)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ===================== MODEL DEFINITION =====================
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), hidden_size)
        c0 = torch.zeros(1, x.size(0), hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# ===================== TRAINING =====================
model = SimpleLSTM(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ===================== INFERENCE =====================
model.eval()
with torch.no_grad():
    test_input = X[-1].unsqueeze(0)
    predicted = []
    for _ in range(50):
        next_pred = model(test_input)
        predicted.append(next_pred.item())
        next_pred_expanded = next_pred.unsqueeze(2)
        test_input = torch.cat((test_input[:, 1:, :], next_pred_expanded), dim=1)

# ===================== METRICS =====================
actual_future = data[len(data) - 50:].flatten()
predicted_arr = np.array(predicted)

# Normalized RMSE
rmse = np.sqrt(np.mean((predicted_arr - actual_future) ** 2))
nrmse = rmse / (actual_future.max() - actual_future.min())

# R² Score
ss_res = np.sum((actual_future - predicted_arr) ** 2)
ss_tot = np.sum((actual_future - np.mean(actual_future)) ** 2)
r2_score = 1 - (ss_res / ss_tot)

print(f"NRMSE: {nrmse:.4f}")
print(f"R² Score: {r2_score:.4f}")

# ===================== VISUALIZATION =====================
plt.figure(figsize=(10, 5))
plt.plot(t[seq_length:], data[seq_length:], label='Actual Data', color='blue')
plt.plot(np.arange(t[-50], t[-50] + 50 * (t[1] - t[0]), t[1] - t[0]),
         predicted, label='Predicted Future', color='red')
plt.axvline(x=t[-50], color='green', linestyle='--', label='Prediction Start')
plt.title(f"LSTM Time Series Prediction | R²: {r2_score:.2f}, NRMSE: {nrmse:.3f}")
plt.xlabel("Time")
plt.ylabel("Vibration Signal")
plt.legend()
plt.grid(True)
plt.show()
