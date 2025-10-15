# Credit Card Fraud Detection Project

This project implements a complete **Machine Learning pipeline** for high-performance credit card fraud detection.  
The focus is on **robust handling of class imbalance** and **optimization for the Area Under the Precision-Recall Curve (AUPRC)**.

---

## Project Goal

To train, optimize, and evaluate classification models capable of identifying rare fraudulent transactions with **high sensitivity (recall)** while maintaining acceptable **precision**.

---

## Dependencies

The following Python libraries are required to run the `fraud_detection.py` script:

- `numpy`  
- `pandas`  
- `scikit-learn`  
- `imbalanced-learn`  
- `matplotlib`  
- `seaborn`

### Installation

```bash
pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn

## 📊 Data Requirement

The script requires the standard **Credit Card Fraud Detection Dataset**.

- **File Name:** `creditcard.csv`  
- **Source:** Available on [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
- **Requirement:** The file must be placed in the **same directory** as the `fraud_detection.py` script for successful execution.

---

## ⚙️ How to Run the Script

The script executes the entire pipeline — from **data loading** and **model training** to **optimization** and **report generation** — automatically.

1. Open your terminal or command prompt.  
2. Navigate to the project directory containing `fraud_detection.py` and `creditcard.csv`.  
3. Run the command:

```bash
python fraud_detection.py



