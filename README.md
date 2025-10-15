Credit Card Fraud Detection:


This project implements a complete Machine Learning pipeline for high-performance credit card fraud detection. The focus is on robust handling of class imbalance and optimization for the Area Under the Precision-Recall Curve (AUPRC).

Project Goal:
To train, optimize, and evaluate classification models capable of identifying rare fraudulent transactions with high sensitivity (recall) while maintaining acceptable precision.

Dependencies:
The following Python libraries are required to run the fraud_detection.py script:-

1.numpy
2.pandas
3.scikit-learn
4.imbalanced-learn
5.matplotlib
6.seaborn

Installation:
pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn

Data Requirement:
The script requires the standard credit card fraud detection dataset.
File Name: creditcard.csv
Source: Available on Kaggle.
Requirement: The file must be placed in the same directory as the fraud_detection.py script for successful execution.

How to Run the Script:
The script executes the entire pipeline, from data loading and model training to optimization and report generation, automatically.
->Open your terminal or command prompt.
->Navigate to the project directory containing final2.py and creditcard.csv.
->Run the command: python fraud_detection.py

Expected Output:
Upon successful completion, the script will:
Print the results of the baseline models and the final optimized model's parameters to the console.
Generate a final report file: final_report_optimized.txt (Text report with test set metrics and best hyperparameters).
Generate a visualization file: fraud_detection_optimized_final.png (Plot summarizing model comparison and final performance metrics).
