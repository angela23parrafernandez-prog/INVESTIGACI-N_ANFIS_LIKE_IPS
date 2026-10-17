
# =====================================================
# IMPORTACIONES
# =====================================================

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    confusion_matrix,
    precision_recall_curve
)

# =====================================================
# REPRODUCIBILIDAD TOTAL
# =====================================================

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# =====================================================
# CONFIGURACIÓN
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET = os.path.join(
    BASE_DIR, "..", "dataset_final", "mqtt_ddos_sampled_labeled.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR, "..", "models", "neuro_fuzzy_tree_model.pkl"
)

FIG_DIR = os.path.join(BASE_DIR, "..", "article_figures")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

FEATURES = ["iat", "pkt_rate", "bytes_rate", "frame.len"]
N_TREES = 30
K_SHARPNESS = 40   # ← Valor óptimo obtenido por análisis de sensibilidad

print(f"\nConfiguración del modelo:")
print(f"N_TREES = {N_TREES}")
print(f"K_SHARPNESS = {K_SHARPNESS}\n")

# =====================================================
# CARGA DE DATOS
# =====================================================

df = pd.read_csv(DATASET)
df["attack"] = (df["label"] != 0).astype(int)

X = df[FEATURES].values
y = df["attack"].values

# =====================================================
# SPLIT 70 / 30
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=RANDOM_STATE
)

# =====================================================
# NORMALIZACIÓN (SIN DATA LEAKAGE)
# =====================================================

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =====================================================
# ENTRENAMIENTO ÁRBOLES BASE
# =====================================================

trees = []
feature_subsets = []

print("Entrenando árboles base...")

for i in range(N_TREES):

    idx = rng.choice(len(X_train), len(X_train), replace=True)
    X_boot = X_train[idx]
    y_boot = y_train[idx]

    selected_features = rng.choice(len(FEATURES), 3, replace=False)
    feature_subsets.append(selected_features)

    tree = DecisionTreeClassifier(
        max_depth=10,
        min_samples_leaf=10,
        random_state=RANDOM_STATE + i
    )

    tree.fit(X_boot[:, selected_features], y_boot)
    trees.append(tree)

print("Árboles entrenados.\n")

# =====================================================
# FUNCIÓN DIFUSA
# =====================================================

def sigmoid_membership(x, threshold, sharpness=K_SHARPNESS):
    return 1 / (1 + np.exp(-sharpness * (x - threshold)))

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

        prob = mu * right_prob + (1 - mu) * left_prob

        node = right if mu > 0.5 else left

    return prob

def generate_tree_features(X_input):

    outputs = []

    for x in X_input:
        tree_scores = [
            fuzzy_tree_output(t, x, feature_subsets[i])
            for i, t in enumerate(trees)
        ]
        outputs.append(tree_scores)

    return np.array(outputs)

# =====================================================
# ENTRENAMIENTO MLP
# =====================================================

X_train_trees = generate_tree_features(X_train)
X_test_trees = generate_tree_features(X_test)

mlp = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    solver="adam",
    max_iter=300,
    shuffle=False,
    random_state=RANDOM_STATE
)

mlp.fit(X_train_trees, y_train)

# =====================================================
# EVALUACIÓN
# =====================================================

y_score = mlp.predict_proba(X_test_trees)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_score)
threshold = thresholds[np.argmax(tpr - fpr)]
y_pred = (y_score >= threshold).astype(int)

print("NEURO-FUZZY TREE MODEL – PERFORMANCE\n")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1-score :", f1_score(y_test, y_pred))
print("AUC      :", roc_auc_score(y_test, y_score))
print("Optimal Threshold:", threshold)

# =====================================================
# MATRIZ DE CONFUSIÓN
# =====================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Attack"],
            yticklabels=["Normal", "Attack"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix – Neuro-Fuzzy Model")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "confusion_matrix.png"), dpi=300)
plt.close()

# =====================================================
# CURVA ROC
# =====================================================

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_score):.3f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Neuro-Fuzzy Model")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "roc_curve.png"), dpi=300)
plt.close()

# =====================================================
# CURVA PRECISION-RECALL
# =====================================================

precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_score)

plt.figure(figsize=(6,5))
plt.plot(recall_vals, precision_vals)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve – Neuro-Fuzzy Model")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "precision_recall_curve.png"), dpi=300)
plt.close()

# =====================================================
# GUARDAR MODELO
# =====================================================

joblib.dump(
    {
        "trees": trees,
        "feature_subsets": feature_subsets,
        "mlp": mlp,
        "scaler": scaler,
        "features": FEATURES,
        "threshold": threshold,
        "k_sharpness": K_SHARPNESS
    },
    MODEL_PATH
)

print("\nModelo guardado en:", MODEL_PATH)
print("Figuras guardadas en:", FIG_DIR)


Configuración del modelo:
N_TREES = 30
K_SHARPNESS = 40

Entrenando árboles base...
Árboles entrenados.

NEURO-FUZZY TREE MODEL – PERFORMANCE

Accuracy : 0.8409722222222222
Precision: 0.902538204236899
Recall   : 0.7645
F1-score : 0.8278040123920956
AUC      : 0.909037247777778
Optimal Threshold: 0.4950053505826551

Modelo guardado en: d:\Maestria Parra\Preprocesamiento y entrenamiento\scripts\..\models\neuro_f