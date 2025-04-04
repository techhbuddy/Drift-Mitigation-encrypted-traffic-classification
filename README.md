# Drift Mitigating Self-Learning System for Encrypted Traffic Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-XGBoost%2BRandom%20Forest-orange)
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
| Metric        | Value  |
|--------------|--------|
| Accuracy     | 95.68% |
| Precision    | 0.94   |
| Recall       | 0.96   |

**Advantages:**
- State-of-the-art classification accuracy
- Built-in feature importance analysis
- Efficient parallel processing

**Limitations:**
- Requires careful hyperparameter tuning
- Needs continuous drift monitoring

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
```sh
# Clone repository
git clone https://github.com/yourusername/drift-mitigation.git
cd drift-mitigation

# Create virtual environment
python -m venv .env
source .env/bin/activate  # Windows: .env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

##Core Dependencies:

*   pandas
    
*   numpy
    
*   scikit-learn
    
*   xgboost
    
*   river (for drift detection)
    
*   matplotlib
    

🚀 Usage
--------

### 1\. Training Models:

shCopy

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python train_random_forest.py  python train_xgboost.py   `

### 2\. Monitoring Drift:

shCopy

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python monitor.py \      --model xgboost_model.pkl \      --stream live_traffic.csv \      --threshold 0.15   `

### 3\. Visualization:

shCopy

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python visualize.py --logs drift_results.json   `

**Sample Output:**

Copy

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   [DRIFT ALERT] 2023-11-20 09:45:12  - Confidence drop: 92% → 68%  - Key drifting features: packet_size, protocol  - Action: Incremental retraining initiated   `

📊 Results
----------

**Key Achievements:**

*   Maintained >90% accuracy under drift conditions
    
*   38% reduction in false positives
    
*   72% faster retraining vs full rebuilds
