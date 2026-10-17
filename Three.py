# =====================================================
# IMPORTS
# =====================================================

# Operating system utilities
import os

# Numerical computations
import numpy as np

# Data manipulation
import pandas as pd

# Statistical test
from scipy.stats import kruskal

# Machine learning models
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# Cross-validation
from sklearn.model_selection import StratifiedKFold

# Feature normalization
from sklearn.preprocessing import MinMaxScaler

# Evaluation metric
from sklearn.metrics import roc_auc_score


# =====================================================
# CONFIGURATION
# =====================================================

# Fixed random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Base directory of the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset path
DATASET = os.path.join(
    BASE_DIR, "..", "dataset_final", "mqtt_ddos_sampled_labeled.csv"
)

# Selected features
FEATURES = ["iat", "pkt_rate", "bytes_rate", "frame.len"]

# Number of trees to evaluate
TREE_VALUES = [10, 20, 30, 40]

# Number of cross-validation folds
N_SPLITS = 3


# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv(DATASET)

# Convert labels to binary classification
# 0 = normal traffic
# 1 = attack traffic
df["attack"] = (df["label"] != 0).astype(int)

# Feature matrix
X = df[FEATURES].values

# Target vector
y = df["attack"].values


# =====================================================
# STRATIFIED CROSS-VALIDATION
# =====================================================

kf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

# Dictionary to store results for each tree configuration
results = {}

for N_TREES in TREE_VALUES:

    print("\nEvaluating N_TREES =", N_TREES)

    auc_scores = []

    for train_idx, test_idx in kf.split(X, y):

        # Train/test split
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Feature normalization
        scaler = MinMaxScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        tree_outputs_train = []
        tree_outputs_test = []

        # =====================================================
        # TRAIN TREE ENSEMBLE
        # =====================================================

        for i in range(N_TREES):

            tree = DecisionTreeClassifier(
                max_depth=8,
                min_samples_leaf=10,
                random_state=RANDOM_STATE + i
            )

            tree.fit(X_train, y_train)

            # Tree probability outputs used as meta-features
            tree_outputs_train.append(tree.predict_proba(X_train)[:, 1])
            tree_outputs_test.append(tree.predict_proba(X_test)[:, 1])

        # Stack tree outputs as input features for the MLP
        X_train_trees = np.column_stack(tree_outputs_train)
        X_test_trees = np.column_stack(tree_outputs_test)

        # =====================================================
        # META-CLASSIFIER (MLP)
        # =====================================================

        mlp = MLPClassifier(
            hidden_layer_sizes=(32, 16),
            max_iter=150,
            random_state=RANDOM_STATE
        )

        mlp.fit(X_train_trees, y_train)

        y_score = mlp.predict_proba(X_test_trees)[:, 1]

        auc = roc_auc_score(y_test, y_score)

        auc_scores.append(auc)

    results[N_TREES] = auc_scores


# =====================================================
# RESULTS SUMMARY
# =====================================================

print("\nRESULTS")

for trees, scores in results.items():

    print(
        "Trees:", trees,
        "Mean AUC:", np.mean(scores),
        "Scores:", scores
    )


# =====================================================
# STATISTICAL TEST
# =====================================================

# Non-parametric Kruskal-Wallis test
# Used to determine whether the number of trees
# significantly affects model performance

stat, p_value = kruskal(*results.values())

print("\nKruskal-Wallis Test")
print("Statistic:", stat)
print("p-value:", p_value)


undled\libs\debugpy\launcher' '56269' '--' 'D:\Maestria Parra\Preprocesamiento y entrenamiento\scripts\arbol.py' 

Evaluando N_TREES = 10

Evaluando N_TREES = 20

Evaluando N_TREES = 30

Evaluando N_TREES = 40

RESULTADOS
Trees: 10 Mean AUC: 0.8971149214708333 Scores: [0.8970855920875, 0.8967945908249999, 0.8974645815]
Trees: 20 Mean AUC: 0.8971144008624999 Scores: [0.8970851862124999, 0.8967940403375, 0.8974639760374999]       
Trees: 30 Mean AUC: 0.8971155169458332 Scores: [0.8970858124, 0.896794334225, 0.8974664042125]
Trees: 40 Mean AUC: 0.8971154082999999 Scores: [0.8970853517999999, 0.8967948448374999, 0.8974660282624999]    

Kruskal-Wallis Test
Statistic: 0.7435897435897516        
p-value: 0.8629064020333611
PS D:\Maestria Parra\Preprocesamiento y entrenamiento>