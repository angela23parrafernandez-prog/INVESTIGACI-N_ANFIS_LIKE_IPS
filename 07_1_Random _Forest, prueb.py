
# ======================================================
# IMPORTS
# ======================================================

# Numerical operations
import numpy as np

# Data manipulation
import pandas as pd

# File system path handling
import os

# Dataset splitting and stratified cross-validation
from sklearn.model_selection import train_test_split, StratifiedKFold

# Multilayer Perceptron classifier
from sklearn.neural_network import MLPClassifier

# Feature normalization
from sklearn.preprocessing import MinMaxScaler

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)

# Ensure reproducibility
np.random.seed(42)


# ======================================================
# LOAD DATASET
# ======================================================

# Obtain the absolute path of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the labeled dataset generated in the preprocessing pipeline
DATASET = os.path.join(BASE_DIR, "..", "dataset_final", "mqtt_ddos_sampled_labeled.csv")

# Load dataset
df = pd.read_csv(DATASET)

# Convert multi-class labels into binary classification
# 0 = normal traffic
# 1 = attack traffic
df["attack"] = (df["label"] != 0).astype(int)

# Selected input features
FEATURES = ["iat", "pkt_rate", "bytes_rate", "frame.len"]

# Feature matrix
X = df[FEATURES].values

# Target vector
y = df["attack"].values


# ======================================================
# FEATURE SCALING (REQUIRED FOR MLP)
# ======================================================

# Normalize features to the [0,1] range
# Neural networks require normalized inputs for stable training
scaler = MinMaxScaler()
X = scaler.fit_transform(X)


# ======================================================
# HOLD-OUT SPLIT (70/30)
# ======================================================

# Split dataset into training and testing subsets
# Stratification preserves the original class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

# Define MLP architecture
model = MLPClassifier(
    hidden_layer_sizes=(32, 16),  # two hidden layers
    activation="tanh",            # activation function
    solver="adam",                # optimization algorithm
    alpha=1e-4,                   # L2 regularization parameter
    max_iter=600,
    random_state=42
)

# Train the neural network
model.fit(X_train, y_train)

# Predicted probabilities for the attack class
y_score = model.predict_proba(X_test)[:, 1]

# Compute ROC curve to determine optimal decision threshold
fpr, tpr, thresholds = roc_curve(y_test, y_score)

# Youden index maximizes sensitivity - false positive rate
threshold = thresholds[np.argmax(tpr - fpr)]

# Final binary predictions
y_pred = (y_score >= threshold).astype(int)

# Hold-out evaluation results
print("\nHOLD-OUT RESULTS\n")
print("AUC:", roc_auc_score(y_test, y_score))
print("F1 :", f1_score(y_test, y_pred))


# ======================================================
# STRATIFIED 5-FOLD CROSS VALIDATION
# ======================================================

# Stratified k-fold ensures class balance in each fold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

auc_scores = []
f1_scores = []

# Iterate through folds
for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):

    # Manual split using fold indices
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    # Initialize a new model for each fold
    model_cv = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="tanh",
        solver="adam",
        alpha=1e-4,
        max_iter=600,
        random_state=42
    )

    # Train model
    model_cv.fit(X_tr, y_tr)

    # Predicted probabilities
    y_score_cv = model_cv.predict_proba(X_te)[:, 1]

    # Compute ROC curve for threshold selection
    fpr_cv, tpr_cv, thresholds_cv = roc_curve(y_te, y_score_cv)

    # Threshold selection using Youden index
    threshold_cv = thresholds_cv[np.argmax(tpr_cv - fpr_cv)]

    # Binary predictions
    y_pred_cv = (y_score_cv >= threshold_cv).astype(int)

    # Evaluation metrics
    auc = roc_auc_score(y_te, y_score_cv)
    f1 = f1_score(y_te, y_pred_cv)

    auc_scores.append(auc)
    f1_scores.append(f1)

    print(f"Fold {fold} -> AUC: {auc:.4f}, F1: {f1:.4f}")


# Cross-validation summary
print("\nSTRATIFIED 5-FOLD RESULTS")
print("Mean AUC:", np.mean(auc_scores))
print("Std AUC :", np.std(auc_scores))
print("Mean F1 :", np.mean(f1_scores))
print("Std F1  :", np.std(f1_scores))


# ======================================================
# BOOTSTRAP CONFIDENCE INTERVALS (95%)
# ======================================================

def bootstrap_ci(y_true, y_score, y_pred, n_bootstrap=1000, seed=42):
    """
    Compute 95% confidence intervals for AUC and F1
    using bootstrap resampling.
    """

    rng = np.random.RandomState(seed)

    auc_values = []
    f1_values = []

    n = len(y_true)

    # Bootstrap resampling with replacement
    for _ in range(n_bootstrap):

        indices = rng.choice(np.arange(n), size=n, replace=True)

        # Ensure both classes are present
        if len(np.unique(y_true[indices])) < 2:
            continue

        auc_values.append(
            roc_auc_score(y_true[indices], y_score[indices])
        )

        f1_values.append(
            f1_score(y_true[indices], y_pred[indices])
        )

    # Compute percentile-based confidence intervals
    auc_ci = (
        np.percentile(auc_values, 2.5),
        np.percentile(auc_values, 97.5)
    )

    f1_ci = (
        np.percentile(f1_values, 2.5),
        np.percentile(f1_values, 97.5)
    )

    return auc_ci, f1_ci


# Compute bootstrap confidence intervals using hold-out predictions
auc_ci, f1_ci = bootstrap_ci(y_test, y_score, y_pred)

print("\nBOOTSTRAP 95% CONFIDENCE INTERVALS")
print(f"AUC 95% CI: {auc_ci[0]:.4f} – {auc_ci[1]:.4f}")
print(f"F1  95% CI: {f1_ci[0]:.4f} – {f1_ci[1]:.4f}")



 _Forest, prueb.py"

HOLD-OUT RESULTS

AUC: 0.8805962873919753
F1 : 0.8029208611147427
Fold 1 -> AUC: 0.8820, F1: 0.8031
Fold 2 -> AUC: 0.8832, F1: 0.8045
Fold 3 -> AUC: 0.8810, F1: 0.8027
Fold 4 -> AUC: 0.8818, F1: 0.8037
Fold 5 -> AUC: 0.8815, F1: 0.8029

STRATIFIED 5-FOLD RESULTS
Mean AUC: 0.8819102521875001
Std AUC : 0.0007142507456012647
Mean F1 : 0.8033829549421518
Std F1  : 0.0006597728181761309

BOOTSTRAP 95% CONFIDENCE INTERVALS
AUC 95% CI: 0.8796 – 0.8816
F1  95% CI: 0.8014 – 0.8043
PS E:\Maestria Parra\Preprocesamiento y entrenamiento>