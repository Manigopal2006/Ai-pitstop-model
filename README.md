# Ai-pitstop-model

check branch 1 latest patch
🏎️ F1 Pit Strategy Predictor – ML + Deep Learning on Apple Silicon

This project predicts tire wear, pit stop chances, and tire degradation in Formula 1 races using both traditional machine learning models and a neural network optimized for Apple’s Metal backend (M1/M2).
🚀 Features
🧠 Hybrid approach using RandomForestClassifier, GradientBoostingRegressor, and a custom Keras neural net.
⚙️ Intelligent preprocessing with label encoding, scaling, and KMeans clustering.
🔥 Built-in support for Apple Silicon acceleration using TensorFlow + Metal.
📈 Evaluation using R², RMSE, Accuracy, F1 Score & classification report.
📦 Output: CSV file containing predictions for tire wear, pit stop likelihood, and degradation percent.
🛠️ Tech Stack
Python, Pandas, NumPy, Scikit-Learn, TensorFlow
ML Models: Gradient Boosting, Random Forest, KMeans Clustering
DL Framework: Keras (TensorFlow backend)
Hardware: M1/M2 with GPU/NPU acceleration
📂 Output
Generates a results.csv with predicted race metrics per lap
