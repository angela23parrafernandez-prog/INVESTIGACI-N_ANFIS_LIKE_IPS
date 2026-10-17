# =====================================================
# IMPORTS
# =====================================================

# Operating system utilities for file path management
import os

# Data manipulation and analysis
import pandas as pd

# Numerical operations
import numpy as np

# Plotting library for visualization
import matplotlib.pyplot as plt

# Decision tree classifier
from sklearn.tree import DecisionTreeClassifier

# Function for splitting dataset into training and testing subsets
from sklearn.model_selection import train_test_split

# Performance evaluation metric (Area Under ROC Curve)
from sklearn.metrics import roc_auc_score

# Feature normalization
from sklearn.preprocessing import MinMaxScaler


# =====================================================
# CONFIGURATION
# =====================================================

# Random seed for reproducibility
RANDOM_STATE = 42

# Absolute path of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the processed dataset used in experiments
DATASET = os.path.join(
    BASE_DIR, "..", "dataset_final", "mqtt_ddos_sampled_labeled.csv"
)

# Directory where generated figures will be stored
FIG_DIR = os.path.join(BASE_DIR, "..", "article_figures")

# Create directory if it does not exist
os.makedirs(FIG_DIR, exist_ok=True)

# Selected features used for training the model
FEATURES = ["iat", "pkt_rate", "bytes_rate", "frame.len"]


# =====================================================
# LOAD DATASET
# =====================================================

# Read the dataset
df = pd.read_csv(DATASET)

# Convert multi-class labels into a binary classification problem
# 0 = normal traffic
# 1 = attack traffic
df["attack"] = (df["label"] != 0).astype(int)

# Feature matrix
X = df[FEATURES].values

# Target vector
y = df["attack"].values


# =====================================================
# TRAIN / TEST SPLIT (70% TRAIN – 30% TEST)
# =====================================================

# Split the dataset while preserving class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=RANDOM_STATE
)


# =====================================================
# FEATURE NORMALIZATION
# =====================================================

# Normalize features to the range [0,1]
# Although decision trees do not strictly require scaling,
# it is applied to maintain experimental consistency
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# =====================================================
# TREE DEPTH SENSITIVITY ANALYSIS
# =====================================================

# Candidate depth values to evaluate
depth_values = [4,6,8,10,12,14,16,18,20,22,30,40]

# Dictionary to store AUC results for each depth
results = {}

print("\nEvaluating tree depth sensitivity...\n")

# Train and evaluate a decision tree for each depth value
for depth in depth_values:

    # Define decision tree model
    tree = DecisionTreeClassifier(
        max_depth=depth,
        min_samples_leaf=10,
        random_state=RANDOM_STATE
    )

    # Train model
    tree.fit(X_train, y_train)

    # Predict attack probability
    y_score = tree.predict_proba(X_test)[:,1]

    # Compute AUC performance
    auc = roc_auc_score(y_test, y_score)

    # Store result
    results[depth] = auc


# =====================================================
# PRINT RESULTS
# =====================================================

print("AUC RESULTS\n")

# Display AUC obtained for each tree depth
for depth, auc in results.items():
    print(f"max_depth = {depth}  ->  AUC = {auc:.4f}")


# =====================================================
# MODEL SELECTION (1% PERFORMANCE RULE)
# =====================================================

# Determine maximum AUC obtained
max_auc = max(results.values())

# Select all depths within 99% of the best AUC
candidate_depths = [
    d for d,a in results.items()
    if a >= max_auc * 0.99
]

# Choose the smallest depth among candidates
# (lower complexity model)
optimal_depth = min(candidate_depths)

print("\nMax AUC:", max_auc)
print("Depths within 1% of maximum:", candidate_depths)
print("Selected depth (minimum complexity):", optimal_depth)


# =====================================================
# GENERATE FIGURE FOR THE ARTICLE
# =====================================================

# Convert dictionary to lists
depths = list(results.keys())
aucs = list(results.values())

# Create plot
plt.figure(figsize=(6,4))

# Plot AUC values for each depth
plt.plot(depths, aucs, marker="o")

# Mark optimal depth with vertical line
plt.axvline(optimal_depth, linestyle="--")

# Axis labels
plt.xlabel("Tree Depth")
plt.ylabel("AUC")

# Figure title
plt.title("Sensitivity Analysis of Tree Depth")

# Adjust layout
plt.tight_layout()

# Save figure with publication quality resolution
plt.savefig(
    os.path.join(FIG_DIR, "depth_sensitivity_auc.png"),
    dpi=300
)

# Display plot
plt.show()



esamiento y entrenamiento\scripts\max_dept.py' 

Evaluando profundidad del árbol...

RESULTADOS AUC

max_depth = 4  ->  AUC = 0.8396
max_depth = 6  ->  AUC = 0.8806
max_depth = 8  ->  AUC = 0.8968
max_depth = 10  ->  AUC = 0.9045
max_depth = 12  ->  AUC = 0.9081
max_depth = 14  ->  AUC = 0.9095
max_depth = 16  ->  AUC = 0.9095
max_depth = 18  ->  AUC = 0.9083
max_depth = 20  ->  AUC = 0.9071
max_depth = 22  ->  AUC = 0.9063     
max_depth = 30  ->  AUC = 0.9043     
max_depth = 40  ->  AUC = 0.9036     

Max AUC: 0.9095401498302469
Depths dentro del 1%: [10, 12, 14, 16, 18, 20, 22, 30, 40]
Depth seleccionado (menor complejidad): 10
