# ======================================================
# IMPORTS
# ======================================================

# Numerical operations
import numpy as np

# Data manipulation and analysis
import pandas as pd

# Operating system utilities for file paths
import os

# Data splitting for hold-out and stratified cross-validation
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

# Ensure reproducibility of random operations
np.random.seed(42)


# ======================================================
# LOAD DATASET
# ======================================================

# Obtain the absolute path of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the labeled dataset generated in the previous pipeline stage
DATASET = os.path.join(BASE_DIR, "..", "dataset_final", "mqtt_ddos_sampled_labeled.csv")

# Load dataset into a pandas dataframe
df = pd.read_csv(DATASET)

# Convert multi-class labels into a binary classification problem
# 0 = normal traffic
# 1 = attack traffic
df["attack"] = (df["label"] != 0).astype(int)

# Selected features used for training the model
FEATURES = ["iat", "pkt_rate", "bytes_rate", "frame.len"]

# Feature matrix
X = df[FEATURES].values

# Target vector
y = df["attack"].values


# ======================================================
# FEATURE SCALING
# ======================================================

# Neural networks are sensitive to feature scale
# Therefore, features are normalized to the range [0,1]
scaler = MinMaxScaler()
X = scaler.fit_transform(X)


# ======================================================
# HOLD-OUT SPLIT (70/30)
# ======================================================

# Split dataset into training and testing subsets
# Stratification preserves the class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

# Define the MLP architecture
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

# Obtain predicted probabilities for the attack class
y_score = model.predict_proba(X_test)[:, 1]

# Compute ROC curve to determine an optimal decision threshold
fpr, tpr, thresholds = roc_curve(y_test, y_score)

# Youden index maximizes sensitivity - false positive rate
threshold = thresholds[np.argmax(tpr - fpr)]

# Apply the selected threshold to obtain binary predictions
y_pred = (y_score >= threshold).astype(int)

# Display hold-out results
print("\nHOLD-OUT RESULTS\n")
print("AUC:", roc_auc_score(y_test, y_score))
print("F1 :", f1_score(y_test, y_pred))


# ======================================================
# STRATIFIED 5-FOLD CROSS-VALIDATION
# ======================================================

# Create stratified k-fold object
# This ensures each fold maintains class balance
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

auc_scores = []
f1_scores = []

# Iterate over folds
for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):

    # Manual split based on fold indices
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    # Create a new model instance for each fold
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

    # Obtain predicted probabilities
    y_score_cv = model_cv.predict_proba(X_te)[:, 1]

    # Compute ROC curve for threshold selection
    fpr_cv, tpr_cv, thresholds_cv = roc_curve(y_te, y_score_cv)

    # Select threshold using Youden index
    threshold_cv = thresholds_cv[np.argmax(tpr_cv - fpr_cv)]

    # Convert probabilities into binary predictions
    y_pred_cv = (y_score_cv >= threshold_cv).astype(int)

    # Compute metrics
    auc = roc_auc_score(y_te, y_score_cv)
    f1 = f1_score(y_te, y_pred_cv)

    auc_scores.append(auc)
    f1_scores.append(f1)

    print(f"Fold {fold} -> AUC: {auc:.4f}, F1: {f1:.4f}")


# Print cross-validation summary statistics
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
    Computes 95% confidence intervals for AUC and F1
    using bootstrap resampling.
    """

    rng = np.random.RandomState(seed)

    auc_values = []
    f1_values = []

    n = len(y_true)

    # Perform bootstrap resampling
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

    # Compute 95% percentile confidence intervals
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




entrenamiento\scripts\06_1Entre_MLPprueba.py' 

HOLD-OUT RESULTS

AUC: 0.8416821268518518
F1 : 0.7588243756220293
Fold 1 -> AUC: 0.8465, F1: 0.7598
Fold 2 -> AUC: 0.8506, F1: 0.7662
Fold 3 -> AUC: 0.8449, F1: 0.7789
Fold 4 -> AUC: 0.8511, F1: 0.7685
Fold 5 -> AUC: 0.8493, F1: 0.7619

STRATIFIED 5-FOLD RESULTS
Mean AUC: 0.8484829497361112
Std AUC : 0.002393769641749331
Mean F1 : 0.7670510126135297
Std F1  : 0.006651521139548287       

BOOTSTRAP 95% CONFIDENCE INTERVALS   
AUC 95% CI: 0.8405 – 0.8430
F1  95% CI: 0.7573 – 0.7604
PS D:\Maestria Parra\Preprocesamiento y entrenamiento>