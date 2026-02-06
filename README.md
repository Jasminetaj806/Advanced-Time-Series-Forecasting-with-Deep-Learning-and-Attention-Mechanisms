


📌 Project Overview
This project implements an advanced deep learning model with an Attention Mechanism for time series forecasting.
A synthetic time series dataset containing trend, multiple seasonal patterns, and Gaussian noise is programmatically generated.
An LSTM-based neural network with a custom Attention layer is built from scratch using TensorFlow/Keras and compared with baseline models.

🎯 Project Objectives
Generate complex synthetic time series data
Build LSTM + Attention forecasting model from scratch
Train and evaluate using MAE, RMSE, and MAPE
Implement baseline models (SARIMA and MLP)
Visualize attention weights for model interpretability
Provide modular and well-documented code
🧠 Model Architecture
LSTM layer to capture temporal dependencies
Custom Attention layer to focus on relevant past timesteps
Dense output layer for prediction
The attention mechanism improves both forecasting performance and interpretability.

📂 Project Structure
Advanced_TS_Project_Forecasting

data_generator.py - Synthetic dataset generation

model.py - LSTM + Attention model

train.py - Model training

evaluate.py - Deep model evaluation (MAE, RMSE, MAPE)

baselines.py - SARIMA and MLP baseline models

attention_visualize.py - Attention heatmap visualization

report.txt - Project report

requirements.txt - Required libraries

README.txt / README.md - Project documentation

Installation

Install required libraries using

pip install -r requirements.txt
▶️ How to Run Step 1: Generate Dataset

python data_generator.py
Step 2: Train Deep Learning Model

python train.py
Step 3: Evaluate Deep Model

python evaluate.py
Step 4: Run Baseline Models

python baselines.py
Step 5: Visualize Attention Weights

python attention_visualize.py
Evaluation Metrics

Models are evaluated using:

Mean Absolute Error (MAE) Root Mean Squared Error (RMSE) Mean Absolute Percentage Error (MAPE)

Attention Visualization

The attention heatmap shows which past time steps the model focuses on while making predictions. This improves interpretability of the forecasting process.

Conclusion

The attention-based LSTM model successfully captures temporal dependencies in complex synthetic data and provides reliable forecasting performance along with interpretability through attention weights.



 
