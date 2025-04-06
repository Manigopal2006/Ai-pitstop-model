import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Hide TF logs

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score, classification_report
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

#  Check if using Apple ML Compute backend (Metal)
print(f"TensorFlow Device: {tf.config.list_logical_devices()}")

# Load Data
train_df = pd.read_csv("/Users/kakkirenimanigopal/Downloads/race_train.csv")
test_df = pd.read_csv("/Users/kakkirenimanigopal/Downloads/race_test.csv")
lap_test = test_df['Lap']

# Preprocessing
combined = pd.concat([train_df.drop(columns=['Tire_Wear', 'Will_Pit_Next_Lap', 'Degradation_Percent']), test_df])
categoricals = ['Track_Condition', 'Tire_Type', 'Driver_Aggressiveness', 'Team', 'Weather_Trend']
encoders = {}
for col in categoricals:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col])
    encoders[col] = le

# Feature Engineering: KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=0)
combined['Pit_Cluster'] = kmeans.fit_predict(combined)

# Scaling
scaler = StandardScaler()
scaled = scaler.fit_transform(combined)
X_train = scaled[:len(train_df)]
X_test = scaled[len(train_df):]

# Targets
y_tire = train_df['Tire_Wear']
y_pit = train_df['Will_Pit_Next_Lap']
y_deg = train_df['Degradation_Percent']

# Traditional ML Models
model_tire = GradientBoostingRegressor(random_state=42)
model_pit = RandomForestClassifier(random_state=42)
model_deg = GradientBoostingRegressor(random_state=42)

model_tire.fit(X_train, y_tire)
model_pit.fit(X_train, y_pit)
model_deg.fit(X_train, y_deg)

# ✅ Neural Network (Runs on M1 NPU/GPU via Metal)
nn_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1)
])
nn_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Fit on M1 Accelerated Backend
nn_model.fit(X_train, y_tire, epochs=20, batch_size=32, verbose=1)

# Predict
pred_tire = model_tire.predict(X_test)
pred_pit = model_pit.predict(X_test)
pred_deg = model_deg.predict(X_test)

# Evaluation
train_pred_tire = model_tire.predict(X_train)
r2_tire = r2_score(y_tire, train_pred_tire)
mse_tire = mean_squared_error(y_tire, train_pred_tire)
rmse_tire = np.sqrt(mse_tire)

train_pred_pit = model_pit.predict(X_train)
acc_pit = accuracy_score(y_pit, train_pred_pit)
f1_pit = f1_score(y_pit, train_pred_pit)

train_pred_deg = model_deg.predict(X_train)
r2_deg = r2_score(y_deg, train_pred_deg)
mse_deg = mean_squared_error(y_deg, train_pred_deg)
rmse_deg = np.sqrt(mse_deg)

print("\n🔧 Model Evaluation on Training Set:")
print(f" Tire Wear → R²: {r2_tire:.4f}, MSE: {mse_tire:.4f}, RMSE: {rmse_tire:.4f}")
print(f" Will Pit → Accuracy: {acc_pit:.4f}, F1 Score: {f1_pit:.4f}")
print(f"Degradation → R²: {r2_deg:.4f}, MSE: {mse_deg:.4f}, RMSE: {rmse_deg:.4f}")
print("\nClassification Report for Pit Stop Prediction:\n")
print(classification_report(y_pit, train_pred_pit))

# Submission
submission = pd.DataFrame({
    'Lap': lap_test,
    'Pred_Tire_Wear': pred_tire,
    'Pred_Will_Pit_Next_Lap': pred_pit,
    'Pred_Degradation_Percent': pred_deg
})
submission.to_csv("results.csv", index=False)
print("✅ Submission saved as: results.csv")
