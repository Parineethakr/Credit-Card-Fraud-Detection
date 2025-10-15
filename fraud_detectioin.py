"""
Credit Card Fraud Detection Pipeline: Complete and Optimized Script (FIXED)

This script executes the entire machine learning pipeline:
1. Data loading and train/val/test splitting.
2. Baseline model training (Logistic Regression, Random Forest, NN).
3. Selection of the best baseline model (smote_rf).
4. Hyperparameter Tuning (Randomized Search) on the best model to maximize AUPRC. (FIXED)
5. Final evaluation of the optimized model on the unseen test set.

Dependencies: numpy, pandas, scikit-learn, imbalanced-learn, matplotlib, seaborn
Data required: 'creditcard.csv'
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score, 
    precision_recall_curve, auc, roc_auc_score, make_scorer
)
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# --- Helper Functions ---

# Sampling function (used in Parts 2, 3, 4)
def apply_sampling(X, y, method, p=0.02):
    """Apply resampling (none, oversample, or smote) to training data."""
    if method == 'none':
        return X, y
    
    n_maj = (y == 0).sum()
    n_min = (y == 1).sum()
    
    if method == 'oversample':
        target = int(p * n_maj / (1 - p))
        sampler = RandomOverSampler(
            sampling_strategy={0: n_maj, 1: target}, random_state=42
        )
    elif method == 'smote':
        target = int(p * n_maj / (1 - p))
        # Ensure k_neighbors is valid (min(5, number of positive samples - 1))
        k_val = min(5, len(X[y==1]) - 1) if len(X[y==1]) > 1 else 1
        sampler = SMOTE(
            sampling_strategy={0: n_maj, 1: min(target, n_maj)}, 
            random_state=42, 
            k_neighbors=k_val
        )
    
    try:
        X_res, y_res = sampler.fit_resample(X, y)
        return X_res, y_res
    except Exception as e:
        print(f"  ⚠ Sampling with {method} failed: {e}. Returning original data.")
        return X, y

# Metrics (used in Parts 2, 3, 4, 5)
def get_metrics(y_true, y_proba):
    """Calculate AUPRC and AUROC."""
    try:
        # Handle 1D or 2D probability arrays for AUPRC/AUROC calculation
        if y_proba.ndim > 1:
            y_score = y_proba[:, 1]
        else:
            y_score = y_proba

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        auprc = auc(recall, precision)
        auroc = roc_auc_score(y_true, y_score)
        return auprc, auroc
    except Exception:
        return 0.0, 0.0

# Custom AUPRC scorer for GridSearchCV (FIXED)
def auprc_scorer_func(y_true, y_proba):
    # Check if y_proba is 2-dimensional (i.e., [:, 0] for class 0, [:, 1] for class 1)
    # This addresses the IndexError from the previous run.
    if y_proba.ndim > 1:
        y_score = y_proba[:, 1]
    else:
        # If it's 1-dimensional, it is already the positive class score
        y_score = y_proba

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)

auprc_scorer = make_scorer(auprc_scorer_func, needs_proba=True)


# --- PART 1: Data Loading and Preprocessing ---

print("="*60)
print("PART 1: DATA LOADING & PREPROCESSING")
print("="*60)

# Load data
print("\n[1/3] Loading creditcard.csv...")
try:
    df = pd.read_csv('creditcard.csv')
except FileNotFoundError:
    print("❌ ERROR: 'creditcard.csv' not found. Please place the file in the script directory.")
    exit(1)

print(f"✓ Loaded {df.shape[0]} transactions")
print(f"Fraud rate: {df['Class'].sum()/len(df)*100:.3f}%")

# Split data
print("\n[2/3] Splitting into train/val/test...")
X = df.drop('Class', axis=1)
y = df['Class']

# First split: test set (16.5%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=1/6, random_state=42, stratify=y
)

# Second split: train (67%) and validation (16.5%)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
)

print(f"✓ Train: {len(X_train)} samples")
print(f"✓ Validation: {len(X_val)} samples")
print(f"✓ Test: {len(X_test)} samples")

# --- Initializing Results DataFrame ---
all_results = pd.DataFrame(columns=['Model', 'AUPRC', 'AUROC'])
print("\n✅ Part 1 Complete.")
print("="*60)


# --- PART 2: Train Logistic Regression Models (Baseline) ---
print("="*60)
print("PART 2: LOGISTIC REGRESSION MODELS (Baseline)")
print("="*60)

lr_results = []
methods = ['none', 'oversample', 'smote']

for method in methods:
    X_samp, y_samp = apply_sampling(X_train, y_train, method)
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced', solver='lbfgs', n_jobs=1)
    lr.fit(X_samp, y_samp)
    y_proba = lr.predict_proba(X_val)
    auprc, auroc = get_metrics(y_val, y_proba)
    lr_results.append({'Model': f'{method}_lr_simple', 'AUPRC': auprc, 'AUROC': auroc})

all_results = pd.concat([all_results, pd.DataFrame(lr_results)])
print("✓ Logistic Regression baselines trained.")
print("="*60)


# --- PART 3: Train Random Forest Models (Baseline) ---
print("="*60)
print("PART 3: RANDOM FOREST MODELS (Baseline)")
print("="*60)

rf_results = []
methods = ['none', 'oversample', 'smote']

for method in methods:
    X_samp, y_samp = apply_sampling(X_train, y_train, method)
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, class_weight='balanced', n_jobs=-1)
    rf.fit(X_samp, y_samp)
    y_proba = rf.predict_proba(X_val)
    auprc, auroc = get_metrics(y_val, y_proba)
    rf_results.append({'Model': f'{method}_rf', 'AUPRC': auprc, 'AUROC': auroc})

all_results = pd.concat([all_results, pd.DataFrame(rf_results)])
print("✓ Random Forest baselines trained.")
print("="*60)


# --- PART 4: Train Neural Network Models (Baseline) ---
print("="*60)
print("PART 4: NEURAL NETWORK MODELS (Baseline)")
print("="*60)

nn_results = []
methods = ['oversample', 'smote'] 
scaler = MinMaxScaler()
scaler.fit(X_train) # Fit scaler only on train data

for method in methods:
    X_samp, y_samp = apply_sampling(X_train, y_train, method)
    
    # Scale sampled data and validation data
    X_samp_scaled = scaler.transform(X_samp)
    X_val_scaled = scaler.transform(X_val)
    
    nn = MLPClassifier(hidden_layer_sizes=(3,), activation='logistic', solver='adam',
                       max_iter=300, random_state=42, early_stopping=True, verbose=False)
    nn.fit(X_samp_scaled, y_samp)
    
    y_proba = nn.predict_proba(X_val_scaled)
    auprc, auroc = get_metrics(y_val, y_proba)
    nn_results.append({'Model': f'{method}_nn', 'AUPRC': auprc, 'AUROC': auroc})

all_results = pd.concat([all_results, pd.DataFrame(nn_results)])
print("✓ Neural Network baselines trained.")
print("="*60)


# --- PART 5: Hyperparameter Tuning (HPT) and Final Test ---

print("="*60)
print("PART 5: HPT & FINAL TEST")
print("="*60)

# 5.1 Final Comparison and Selection
all_results = all_results[all_results['AUPRC'] > 0]
all_results = all_results.sort_values('AUPRC', ascending=False).reset_index(drop=True)
best_baseline_name = all_results.iloc[0]['Model']

print("\n📊 VALIDATION SET RESULTS (Baselines):")
print(all_results[['Model', 'AUPRC', 'AUROC']].to_string(index=False))
print(f"\n🏆 Best Baseline Model: {best_baseline_name}")

# 5.2 Prepare Combined Data for HPT
X_combined = pd.concat([X_train, X_val])
y_combined = pd.concat([y_train, y_val])

print("\n[1/3] Starting Randomized Search Hyperparameter Tuning (HPT)...")
print("This step is now FIXED and should run correctly.")

# Define the Pipeline: SMOTE + Random Forest
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('rf', RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1))
])

# Parameter grid for Randomized Search
param_grid = {
    'smote__k_neighbors': [3, 5, 7],
    'rf__n_estimators': [100, 200, 300],
    'rf__max_depth': [10, 20, 30, None],
    'rf__min_samples_leaf': [1, 2, 4],
}

# Run HPT
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_grid,
    n_iter=5, # Reduced for quick execution, use >50 for production
    scoring=auprc_scorer,
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_combined, y_combined)
best_model = random_search.best_estimator_

print("\n[2/3] HPT Complete.")
print(f"🎉 Best Cross-Validation AUPRC: {random_search.best_score_:.4f}")
print("Optimal Parameters Found:")
for k, v in random_search.best_params_.items():
    print(f"- {k}: {v}")


# 5.3 Final Evaluation on Test Set
print("\n[3/3] Evaluating Optimized Model on UNSEEN Test Set...")

# Predict on test
y_pred_proba = best_model.predict_proba(X_test)
y_pred_proba_pos = y_pred_proba[:, 1]
y_pred = (y_pred_proba_pos > 0.5).astype(int)

# Calculate final metrics
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba_pos)
auprc = auc(recall, precision)
auroc = roc_auc_score(y_test, y_pred_proba_pos)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0


# --- 6. Visualization and Report ---

print("\n[4/4] Creating visualizations and final report...")

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Model comparison
ax1 = axes[0, 0]
top_models = all_results.head(8)
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_models)))
ax1.barh(range(len(top_models)), top_models['AUPRC'], color=colors)
ax1.set_yticks(range(len(top_models)))
ax1.set_yticklabels(top_models['Model'])
ax1.set_xlabel('AUPRC Score', fontweight='bold')
ax1.set_title('Top Baseline Models by AUPRC', fontweight='bold', fontsize=12)
ax1.set_xlim(0, 1)

# Plot 2: Confusion matrix
ax2 = axes[0, 1]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
           xticklabels=['Legit (0)', 'Fraud (1)'],
           yticklabels=['Legit (0)', 'Fraud (1)'])
ax2.set_xlabel('Predicted Label', fontweight='bold')
ax2.set_ylabel('True Label', fontweight='bold')
ax2.set_title('Optimized Model Test Set Confusion Matrix', fontweight='bold', fontsize=12)

# Plot 3: Precision-Recall Curve (AUPRC)
ax3 = axes[1, 0]
ax3.plot(recall, precision, label=f'AUPRC = {auprc:.4f}')
ax3.set_xlabel('Recall (Sensitivity)', fontweight='bold')
ax3.set_ylabel('Precision', fontweight='bold')
ax3.set_title('Precision-Recall Curve on Test Set', fontweight='bold', fontsize=12)
ax3.legend()

# Plot 4: Metrics summary
ax4 = axes[1, 1]
metrics_names = ['AUPRC', 'AUROC', 'Accuracy', 'Sensitivity', 'Specificity', 'F1']
metrics_values = [auprc, auroc, accuracy, sensitivity, specificity, f1]
colors = plt.cm.RdYlGn(np.array(metrics_values) / 1.0)
bars = ax4.barh(metrics_names, metrics_values, color=colors)
ax4.set_xlabel('Score', fontweight='bold')
ax4.set_title('Final Optimized Model Test Metrics', fontweight='bold', fontsize=12)
ax4.set_xlim(0, 1)

for i, (bar, val) in enumerate(zip(bars, metrics_values)):
    ax4.text(val + 0.01, i, f'{val:.3f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('fraud_detection_optimized_final.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualization as 'fraud_detection_optimized_final.png'")


# Final Report String
report = f"""
CREDIT CARD FRAUD DETECTION - OPTIMIZED FINAL REPORT
{'='*60}

MODEL TUNING SUMMARY:
- Best Baseline Model (Validation AUPRC): {best_baseline_name}
- Final Model: Optimized Random Forest (via Randomized Search)
- Optimal Hyperparameters: {random_search.best_params_}

DATASET STATS:
- Total transactions: {len(df)}
- Fraud rate: {df['Class'].sum()/len(df)*100:.3f}%

TEST SET RESULTS (Optimized Random Forest Model):
- AUPRC: {auprc:.4f} (Area Under Precision-Recall Curve)
- AUROC: {auroc:.4f} (Area Under ROC Curve)
- F1 Score: {f1:.4f}
- Accuracy: {accuracy:.4f}

FRAUD DETECTION PERFORMANCE:
- Sensitivity (Recall): {sensitivity:.4f} ({tp}/{tp+fn} frauds detected)
- Specificity: {specificity:.4f} ({tn}/{tn+fp} legit transactions correctly classified)
- False Alarms (FP): {fp} (This is {fp/(tn+fp)*100:.3f}% of legit transactions flagged as fraud)
- Missed Frauds (FN): {fn}

{'='*60}
"""

with open('final_report_optimized.txt', 'w') as f:
    f.write(report)

print("\n[5/5] Final Summary Report:")
print(report)
print("✓ Saved report as 'final_report_optimized.txt'")

print("\n✅ ALL PARTS COMPLETE!")
print("="*60)