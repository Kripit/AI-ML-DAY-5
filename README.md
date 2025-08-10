# Cross-Domain Predictive Maintenance (CD-PM) with LSTM

## 📌 Overview
This project demonstrates a **Long Short-Term Memory (LSTM)** model applied to synthetic vibration-like time series data for predictive maintenance.  
The goal is to forecast potential anomalies and estimate remaining useful life (RUL) of machines.

We combine deep learning with **time-series forecasting** to create a foundation for real-world predictive maintenance systems — scalable for industrial use.

---

## 🚀 Why LSTM?
* Traditional CNNs excel in spatial pattern recognition (like images), but **time-series data** requires remembering patterns over time.
* Simple RNNs suffer from **vanishing gradients** and cannot remember long-term dependencies.
* LSTMs solve this with **memory cells** and **gating mechanisms** to decide what information to keep, forget, and output.

---
## 🛠 How It Works
1. **Data Generation**: Create synthetic vibration signal.
2. **Sequence Preparation**: Slice data into overlapping sequences for time-series forecasting.
3. **Model Architecture**:  
   - LSTM layer for sequential learning  
   - Fully connected layer for prediction  
4. **Training**: Optimize MSE loss with Adam optimizer.
5. **Prediction**: Forecast next time steps and compare with actual signal.
