# ======================================================
# DATASET SAMPLING AND LABELING SCRIPT
# This script assigns class labels based on traffic
# scenarios and performs balanced sampling for ML training.
# ======================================================

# Import pandas for data manipulation
import pandas as pd

# Import os to manage file paths
import os


# ======================================================
# DIRECTORY CONFIGURATION
# ======================================================

# Obtain the absolute path of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the input dataset with engineered features
DATASET_IN = os.path.join(BASE_DIR, "..", "dataset_final", "mqtt_ddos_features.csv")

# Path where the sampled and labeled dataset will be stored
DATASET_OUT = os.path.join(BASE_DIR, "..", "dataset_final", "mqtt_ddos_sampled_labeled.csv")


# ======================================================
# SAMPLING PARAMETERS
# ======================================================

# Maximum number of samples per class
SAMPLES_PER_CLASS = 300_000

# Random seed to ensure reproducibility of sampling
RANDOM_STATE = 42


# ======================================================
# LOAD DATASET
# ======================================================

# Load the dataset containing engineered features
df = pd.read_csv(DATASET_IN)


# ======================================================
# SCENARIO → CLASS LABEL MAPPING
# ======================================================

# Map traffic scenarios to numerical labels used in ML models
# 0 → Normal traffic
# 1 → Basic flooding attack
# 2 → Delayed flooding attack

label_map = {
    "normal": 0,
    "normal_ecg": 0,
    "flooding_basic": 1,
    "flooding_delayed": 2
}

# Create a new column containing the numeric label
df["label"] = df["scenario"].map(label_map)


# ======================================================
# BALANCED SAMPLING BY CLASS
# ======================================================

# List to store sampled data for each class
samples = []

# Iterate over each class label defined in the map
for label_id in label_map.values():

    # Filter dataset by class
    df_class = df[df["label"] == label_id]

    # Perform random sampling up to the defined limit
    samples.append(
        df_class.sample(
            n=min(SAMPLES_PER_CLASS, len(df_class)),  # Avoid sampling more rows than available
            random_state=RANDOM_STATE
        )
    )


# ======================================================
# FINAL DATASET GENERATION
# ======================================================

# Combine sampled data from all classes
sampled_df = pd.concat(samples, ignore_index=True)

# Save the balanced and labeled dataset
sampled_df.to_csv(DATASET_OUT, index=False)


# Display completion message
print("Sampled and labeled dataset successfully generated")