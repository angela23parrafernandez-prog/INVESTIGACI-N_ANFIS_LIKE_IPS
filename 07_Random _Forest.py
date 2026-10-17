# ======================================================
# IMPORTS
# ======================================================

# Numerical operations
import numpy as np

# Tabular data manipulation
import pandas as pd

# Operating system path management
import os

# Save and load trained models
import joblib

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning utilities
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)


# ======================================================
# PATH CONFIGURATION
# ======================================================

# Obtain absolute path of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the labeled dataset generated in the preprocessing pipeline
DATASET = os.path.join(
    BASE_DIR, "..", "dataset_final", "mqtt_ddos_sampled_labeled.csv"
)

# Path where the trained Random Forest model will be stored
MODEL_PATH = os.path.join(
    BASE_DIR, "..", "models", "rf_ips_model.pkl"
)

# Directory where figures for the article will be saved
FIG_DIR = os.path.join(
    BASE_DIR, "..", "article_figures"
)

# Create directory if it does not exist
os.makedirs(FIG_DIR, exist_ok=True)

print("Loading dataset:", DATASET)


# ======================================================
# LOAD DATASET
# ======================================================

# Load dataset
df = pd.read_csv(DATASET)

# Convert multi-class label into binary classification
# 0 = normal traffic
# 1 = attack traffic
df["attack"] = (df["label"] != 0).astype(int)

# Features used for model training
FEATURES = ["iat", "pkt_rate", "bytes_rate", "frame.len"]

# Feature matrix
X = df[FEATURES].values

# Target vector
y = df["attack"].values


# ======================================================
# FEATURE SCALING
# ======================================================

# Random Forest does not require feature scaling.
# However, scaling is applied here to maintain
# consistency with other experimental models.
scaler = MinMaxScaler()
X = scaler.fit_transform(X)


# ======================================================
# TRAIN / TEST SPLIT
# ======================================================

# 70% training
# 30% testing
# Stratification preserves class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    stratify=y,
    random_state=42
)


# ======================================================
# RANDOM FOREST MODEL DEFINITION
# ======================================================

model = RandomForestClassifier(
    n_estimators=200,      # Number of trees in the forest
    max_depth=None,        # Trees grow until no further splits are possible
    min_samples_split=2,   # Minimum samples required to split a node
    min_samples_leaf=1,    # Minimum samples required at leaf nodes
    random_state=42,
    n_jobs=-1              # Use all CPU cores
)

print("Training Random Forest model...")

# Train the model
model.fit(X_train, y_train)


# ======================================================
# MODEL INFERENCE
# ======================================================

# Obtain probability estimates for the attack class
y_score = model.predict_proba(X_test)[:, 1]

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_score)

# Select optimal threshold using the Youden index
threshold = thresholds[np.argmax(tpr - fpr)]

# Convert probabilities into binary predictions
y_pred = (y_score >= threshold).astype(int)


# ======================================================
# PERFORMANCE METRICS
# ======================================================

print("\nRANDOM FOREST PERFORMANCE\n")

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1-score :", f1_score(y_test, y_pred))
print("AUC      :", roc_auc_score(y_test, y_score))
print("Optimal Threshold:", threshold)


# ======================================================
# SAVE TRAINED MODEL
# ======================================================

# Store:
# - trained model
# - scaler
# - feature list
# - optimal decision threshold
joblib.dump(
    {
        "model": model,
        "scaler": scaler,
        "features": FEATURES,
        "threshold": threshold
    },
    MODEL_PATH
)

print("\nModel saved at:", MODEL_PATH)


# ======================================================
# FIGURE 1 – ROC CURVE
# ======================================================

plt.figure(figsize=(6, 5))

# Plot ROC curve
plt.plot(
    fpr,
    tpr,
    label=f"Random Forest (AUC = {roc_auc_score(y_test, y_score):.3f})",
    linewidth=2
)

# Random classifier reference line
plt.plot([0, 1], [0, 1], "--", color="gray")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Random Forest Intrusion Detection")
plt.legend(loc="lower right")
plt.tight_layout()

# Save figure at publication quality (300 dpi)
plt.savefig(
    os.path.join(FIG_DIR, "Fig_ROC_RF.png"),
    dpi=300
)

plt.close()


# ======================================================
# FIGURE 2 – CONFUSION MATRIX
# ======================================================

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(4, 3))

# Plot heatmap
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["Normal", "Attack"],
    yticklabels=["Normal", "Attack"]
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix – Random Forest")
plt.tight_layout()

# Save figure
plt.savefig(
    os.path.join(FIG_DIR, "Fig_CM_RF.png"),
    dpi=300
)

plt.close()

print("\nProcess completed successfully.")
print("Figures saved in:", FIG_DIR)




Training Random Forest model...
labeled.csv
RANDOM FOREST PERFORMANCE

RANDOM FOREST PERFORMANCE

Accuracy : 0.8456972222222222
RANDOM FOREST PERFORMANCE

Accuracy : 0.8456972222222222        
Precision: 0.9294820029678711        
Recall   : 0.7481555555555556        
F1-score : 0.8290194992074118        
AUC      : 0.9085065727932099        
Optimal Threshold: 0.58875

Model saved at: D:\Maestria Parra\Preprocesamiento y entrenamiento\scripts\..\models\rf_ips_model.pkl

Process completed successfully.      
Figures saved in: D:\Maestria Parra\Preprocesamiento y entrenamiento\scripts\..\article_figures
PS D:\Maestria Parra\Preprocesamiento y entrenamiento>