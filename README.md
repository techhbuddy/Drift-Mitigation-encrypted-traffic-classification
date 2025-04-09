# Drift Mitigating Self-Learning System for Encrypted Traffic Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-XGBoost%2BRandom%20Forest%2BKNN%2BMLP%2BSVM%2BHoeffding%20Tree-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## 🎓 Course Project
This project was developed as part of the Course Project at **Vellore Institute of Technology**, guided by **Aswani Kumar Cherukuri**. The system addresses concept drift in encrypted network traffic classification using adaptive machine learning techniques.

## 📖 Introduction
Encrypted traffic classification is vital for:
- Malicious activity detection
- Quality of Service (QoS) maintenance
- Network efficiency optimization

**Key Challenge:** Concept drift degrades model performance as traffic patterns evolve.

## 🧠 Machine Learning Models

### 1. Random Forest
**Description:**  
Random Forest is an ensemble learning method that constructs multiple decision trees and aggregates their predictions to improve accuracy and reduce overfitting. It is highly effective in handling non-linear relationships and high-dimensional data, making it suitable for encrypted traffic classification.

**Performance:**
| Metric        | Value  |
|--------------|--------|
| Accuracy     | 81.6%  |
| Precision    | 0.82   |
| Recall       | 0.81   |

**Advantages:**
- Robust against noise/outliers
- Handles feature interactions well
- Reduces overfitting through ensemble approach

**Limitations:**
- Computationally intensive for large datasets
- Requires drift adaptation mechanisms

### 2. XGBoost (Extreme Gradient Boosting)
**Description:**  
XGBoost is an advanced gradient boosting framework that optimizes model performance through sequential tree building and regularization. It is highly efficient and widely used in classification tasks.

**Performance:**
| Metric                | Value   |
|-----------------------|---------|
| Accuracy (Train)      | 86.12%  |
| Accuracy (Test)       | 78.79%  |
| Pre-drift Accuracy    | 86.58%  |
| Post-drift Accuracy   | 83.11%  |

**Advantages:**
- State-of-the-art classification accuracy
- Built-in feature importance analysis
- Efficient parallel processing

**Limitations:**
- Requires careful hyperparameter tuning
- Needs continuous drift monitoring

### 3. Multi-Layer Perceptron (MLP)
**Description:**  
MLP is a class of feedforward artificial neural network consisting of at least three layers. It uses backpropagation for training and is capable of learning complex patterns.

**Performance:**
| Metric                | Value   |
|-----------------------|---------|
| Accuracy (Train)      | 61.98%  |
| Accuracy (Test)       | 62.42%  |
| Pre-drift Accuracy    | 68.96%  |
| Post-drift Accuracy   | 66.03%  |

**Advantages:**
- Learns non-linear decision boundaries
- Good for pattern recognition

**Limitations:**
- Prone to overfitting
- Requires careful tuning of architecture and hyperparameters

### 4. K-Nearest Neighbors (KNN)
**Description:**  
KNN is a lazy learning algorithm that classifies data based on the majority label among the K-nearest samples in the feature space. A sliding window approach was used in the streaming context.

**Performance:**
| Metric                | Value   |
|-----------------------|---------|
| Accuracy (Train)      | 69.08%  |
| Accuracy (Test)       | 61.71%  |
| Pre-drift Accuracy    | 61.71%  |
| Post-drift Accuracy   | 61.58%  |

**Advantages:**
- Simple and interpretable
- Effective with smaller datasets

**Limitations:**
- Computationally expensive for large data
- Sensitive to irrelevant features and noisy data

### 5. Hoeffding Tree
**Description:**  
Hoeffding Tree is a very fast decision tree learner for streaming data. It uses the Hoeffding bound to determine the number of samples needed to make decisions.

**Performance:**
| Metric                | Value   |
|-----------------------|---------|
| Accuracy (Train)      | 57.62%  |
| Accuracy (Test)       | 62.69%  |
| Pre-drift Accuracy    | 61.04%  |
| Post-drift Accuracy   | 60.90%  |

**Advantages:**
- Designed for streaming data
- Fast and memory-efficient

**Limitations:**
- Less accurate compared to ensemble models
- May require tuning for drift sensitivity

### 6. Support Vector Machine (SVM)
**Description:**  
SVM is a supervised learning model that classifies data by finding the optimal hyperplane that separates data points of different classes. A linear kernel was used in this project.

**Performance:**
| Metric                | Value   |
|-----------------------|---------|
| Accuracy (Train)      | 56.92%  |
| Accuracy (Test)       | 56.34%  |
| Pre-drift Accuracy    | 49.94%  |
| Post-drift Accuracy   | 49.82%  |

**Advantages:**
- Effective in high-dimensional spaces
- Memory efficient

**Limitations:**
- Not ideal for large datasets
- Sensitive to noise and outliers



## 📉 Drift Detection Techniques

To monitor and detect **concept drift** in encrypted traffic data, we utilized three popular streaming drift detection methods from the **River** library:

### 🧪 1. ADWIN (Adaptive Windowing)

- Dynamically adjusts window size based on statistical changes in the data stream.
- Best suited for detecting **gradual drift**.
- Maintains two sub-windows and compares their means to determine if a change has occurred.

### 🔍 2. KSWIN (Kolmogorov–Smirnov WINdowing)

- A **non-parametric** test that compares distributions between recent and past data within a sliding window.
- Effective for identifying **distributional changes**.
- Suitable for detecting both sudden and gradual drifts without strong assumptions about the data.

### ⚠️ 3. Page-Hinkley

- Detects **abrupt/sudden drift** by analyzing the cumulative differences between observed values and their mean.
- Triggers alarms when the average change in performance exceeds a certain threshold.

---

These detectors were applied to **selected traffic features**, and the points where drift was detected were **recorded for further analysis and model retraining**.


## 🛡️ Drift Mitigation Framework

### Detection Mechanisms:
1. **Statistical Process Control (SPC)**  
   Monitors prediction errors over time and triggers retraining when thresholds are exceeded
2. **Adaptive Windowing**  
   Dynamically adjusts training windows to focus on recent data
3. **Ensemble-Based Detection**  
   Uses multiple models to identify distribution shifts

### Adaptation Strategies:
- **Incremental Learning:** Updates models with new data
- **Periodic Retraining:** Scheduled model refreshes
- **Active Learning:** Focuses on most informative samples
- **Feature Re-weighting:** Adjusts feature importance dynamically

## 💻 Installation

### Requirements:
- Python 3.8+
- GPU recommended for training

### Setup:

# Clone repository
git clone https://github.com/yourusername/drift-mitigation.git
cd drift-mitigation

# Create virtual environment
python -m venv .env
source .env/bin/activate  # Windows: .env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# 🔐 Encrypted Traffic Classification with Drift Mitigation

## 📦 Dependencies

Ensure you have the following libraries installed:

- Python 3.8+
- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `matplotlib`
- `river` (for drift detection)

Install all dependencies using:


pip install -r requirements.txt

## 🚀 Usage

### 1. Train the Models

Use the following commands to train the models:


python train_random_forest.py  
python train_xgboost.py

### 2. Evaluate Performance

After training, evaluate the models using the following metrics:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Drift Detection Metrics**  
  (e.g., error rate, distribution shifts)

---

### 3. Run Drift Adaptation

To handle concept drift and retrain models accordingly, run:


python drift_mitigation.py

## ✅ Conclusion

This project introduces a **self-learning system for encrypted traffic classification** that effectively mitigates **concept drift**. By combining **Random Forest** and **XGBoost** with **adaptive retraining** and **drift detection**, we ensure **long-term model reliability**.




    



