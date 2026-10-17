# =====================================================
# IMPORTS
# =====================================================

# Operating system utilities
import os

# Numerical operations
import numpy as np

# Data manipulation
import pandas as pd

# Save and load trained models
import joblib

# Machine learning models
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# Dataset splitting and stratified cross-validation
from sklearn.model_selection import train_test_split, StratifiedKFold

# Feature normalization
from sklearn.preprocessing import MinMaxScaler

# Evaluation metrics
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve
)

# =====================================================
# REPRODUCIBILITY SETTINGS
# =====================================================

# Fix random seeds for reproducibility
RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# =====================================================
# CONFIGURATION
# =====================================================

# Obtain the absolute path of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the labeled dataset
DATASET = os.path.join(
    BASE_DIR, "..", "dataset_final", "mqtt_ddos_sampled_labeled.csv"
)

# Path where the trained model will be stored
MODEL_PATH = os.path.join(
    BASE_DIR, "..", "models", "neuro_fuzzy_tree_model.pkl"
)

# Selected input features
FEATURES = ["iat", "pkt_rate", "bytes_rate", "frame.len"]

# Number of trees used in the ensemble
N_TREES = 30


# =====================================================
# LOAD DATASET
# =====================================================

# Load dataset
df = pd.read_csv(DATASET)

# Convert multi-class label into binary classification
# 0 = normal traffic
# 1 = attack traffic
df["attack"] = (df["label"] != 0).astype(int)

# Feature matrix
X = df[FEATURES].values

# Target vector
y = df["attack"].values


# =====================================================
# TRAINING FUNCTION
# =====================================================

def train_neuro_fuzzy(X_train, y_train):
    """
    Train the hybrid neuro-fuzzy tree ensemble model.

    The model consists of:
    1) An ensemble of decision trees trained on bootstrapped samples
    2) A sigmoidal membership function that smooths decision boundaries
    3) A neural network meta-classifier that aggregates tree outputs
    """

    trees = []
    feature_subsets = []

    # -------------------------------------------------
    # Train ensemble of bootstrapped decision trees
    # -------------------------------------------------

    for i in range(N_TREES):

        # Bootstrap sampling
        idx = rng.choice(len(X_train), len(X_train), replace=True)

        X_boot = X_train[idx]
        y_boot = y_train[idx]

        # Random feature subset (feature bagging)
        selected_features = rng.choice(len(FEATURES), 3, replace=False)

        feature_subsets.append(selected_features)

        # Train decision tree
        tree = DecisionTreeClassifier(
            max_depth=10,
            min_samples_leaf=10,
            random_state=RANDOM_STATE + i
        )

        tree.fit(X_boot[:, selected_features], y_boot)

        trees.append(tree)

    # -------------------------------------------------
    # Sigmoidal membership function
    # -------------------------------------------------

    def sigmoid_membership(x, threshold, sharpness=40):
        """
        Sigmoid function used to smooth decision thresholds,
        producing fuzzy membership values between 0 and 1.
        """
        return 1 / (1 + np.exp(-sharpness * (x - threshold)))

    # -------------------------------------------------
    # Fuzzy decision tree evaluation
    # -------------------------------------------------

    def fuzzy_tree_output(tree, x, selected_features):

        node = 0
        x_sub = x[selected_features]

        while tree.tree_.children_left[node] != -1:

            feature = tree.tree_.feature[node]
            threshold = tree.tree_.threshold[node]

            mu = sigmoid_membership(x_sub[feature], threshold)

            left = tree.tree_.children_left[node]
            right = tree.tree_.children_right[node]

            left_prob = tree.tree_.value[left][0][1]
            right_prob = tree.tree_.value[right][0][1]

            # Fuzzy interpolation between child nodes
            prob = mu * right_prob + (1 - mu) * left_prob

            node = right if mu > 0.5 else left

        return prob

    # -------------------------------------------------
    # Generate tree outputs as meta-features
    # -------------------------------------------------

    def generate_tree_features(X_input):

        outputs = []

        for x in X_input:

            scores = [
                fuzzy_tree_output(t, x, feature_subsets[i])
                for i, t in enumerate(trees)
            ]

            outputs.append(scores)

        return np.array(outputs)

    # Generate tree outputs for training data
    X_train_trees = generate_tree_features(X_train)

    # -------------------------------------------------
    # Neural network meta-classifier
    # -------------------------------------------------

    mlp = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        max_iter=300,
        shuffle=False,
        random_state=RANDOM_STATE
    )

    mlp.fit(X_train_trees, y_train)

    return trees, feature_subsets, mlp, generate_tree_features


# =====================================================
# 1) HOLD-OUT EVALUATION
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    stratify=y,
    random_state=RANDOM_STATE
)

# Normalize features
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train hybrid model
trees, feature_subsets, mlp, generate_tree_features = train_neuro_fuzzy(
    X_train,
    y_train
)

# Generate meta-features for test set
X_test_trees = generate_tree_features(X_test)

# Predict attack probabilities
y_score = mlp.predict_proba(X_test_trees)[:, 1]

# ROC-based threshold selection
fpr, tpr, thresholds = roc_curve(y_test, y_score)

threshold = thresholds[np.argmax(tpr - fpr)]

# Final predictions
y_pred = (y_score >= threshold).astype(int)

auc_hold = roc_auc_score(y_test, y_score)
f1_hold = f1_score(y_test, y_pred)

print("\nHOLD-OUT RESULTS\n")

print("AUC:", auc_hold)
print("F1 :", f1_hold)


# =====================================================
# 2) STRATIFIED 5-FOLD CROSS-VALIDATION
# =====================================================

kf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

auc_scores = []
f1_scores = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):

    X_train_cv = X[train_idx]
    X_test_cv = X[test_idx]

    y_train_cv = y[train_idx]
    y_test_cv = y[test_idx]

    scaler_cv = MinMaxScaler()

    X_train_cv = scaler_cv.fit_transform(X_train_cv)
    X_test_cv = scaler_cv.transform(X_test_cv)

    trees, feature_subsets, mlp, generate_tree_features = train_neuro_fuzzy(
        X_train_cv,
        y_train_cv
    )

    X_test_trees = generate_tree_features(X_test_cv)

    y_score_cv = mlp.predict_proba(X_test_trees)[:, 1]

    fpr_cv, tpr_cv, thresholds_cv = roc_curve(y_test_cv, y_score_cv)

    threshold_cv = thresholds_cv[np.argmax(tpr_cv - fpr_cv)]

    y_pred_cv = (y_score_cv >= threshold_cv).astype(int)

    auc_fold = roc_auc_score(y_test_cv, y_score_cv)
    f1_fold = f1_score(y_test_cv, y_pred_cv)

    print(f"Fold {fold} -> AUC: {auc_fold:.4f}, F1: {f1_fold:.4f}")

    auc_scores.append(auc_fold)
    f1_scores.append(f1_fold)

print("\nSTRATIFIED 5-FOLD RESULTS")

print("Mean AUC:", np.mean(auc_scores))
print("Std AUC :", np.std(auc_scores))

print("Mean F1 :", np.mean(f1_scores))
print("Std F1  :", np.std(f1_scores))


# =====================================================
# 3) BOOTSTRAP CONFIDENCE INTERVALS
# =====================================================

n_bootstrap = 1000

boot_auc = []
boot_f1 = []

rng_boot = np.random.default_rng(RANDOM_STATE)

for _ in range(n_bootstrap):

    idx = rng_boot.choice(len(y_test), len(y_test), replace=True)

    if len(np.unique(y_test[idx])) < 2:
        continue

    boot_auc.append(roc_auc_score(y_test[idx], y_score[idx]))
    boot_f1.append(f1_score(y_test[idx], y_pred[idx]))

auc_ci = np.percentile(boot_auc, [2.5, 97.5])
f1_ci = np.percentile(boot_f1, [2.5, 97.5])

print("\nBOOTSTRAP 95% CONFIDENCE INTERVALS")

print(f"AUC 95% CI: {auc_ci[0]:.4f} – {auc_ci[1]:.4f}")
print(f"F1  95% CI: {f1_ci[0]:.4f} – {f1_ci[1]:.4f}")


# =====================================================
# SAVE MODEL
# =====================================================

joblib.dump(
    {
        "trees": trees,
        "feature_subsets": feature_subsets,
        "mlp": mlp,
        "scaler": scaler,
        "features": FEATURES,
        "threshold": threshold
    },
    MODEL_PATH
)

print("\nModel saved at:", MODEL_PATH)

modelo hibrido propuesto
-win32-x64\bundled\libs\debugpy\launcher' '52409' '--' 'D:\Maestria Parra\Preprocesamiento y entrenamiento\scripts\8_neurodifuso_prueb.py'

HOLD-OUT RESULTS

AUC: 0.909037247777778
F1 : 0.8278040123920956
Fold 1 -> AUC: 0.8955, F1: 0.8206
Fold 2 -> AUC: 0.9048, F1: 0.8260
Fold 3 -> AUC: 0.8999, F1: 0.8208
Fold 4 -> AUC: 0.9017, F1: 0.8221
Fold 5 -> AUC: 0.9017, F1: 0.8215

STRATIFIED 5-FOLD RESULTS
Mean AUC: 0.900711247923611
Std AUC : 0.003033444595954215
Mean F1 : 0.8222043883140033
Std F1  : 0.0019946723822622187

BOOTSTRAP 95% CONFIDENCE INTERVALS
AUC 95% CI: 0.9080 – 0.9100
F1  95% CI: 0.8264 – 0.8292

Modelo guardado en: D:\Maestria Parra\Preprocesamiento y entrenamiento\scripts\..\models\neuro_fuzzy_tree_model.pkl
PS D:\Maestria Parra\Preprocesamiento y entrenamiento> 