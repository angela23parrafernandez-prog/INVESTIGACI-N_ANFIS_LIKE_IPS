# ======================================================
# FEATURE ENGINEERING FOR MQTT DDoS DATASET
# This script processes cleaned CSV files and generates
# a final dataset with engineered features for ML models.
# ======================================================

# Import pandas for data manipulation and transformation
import pandas as pd

# Allows searching files using patterns (e.g., *.csv)
import glob

# Provides utilities for working with file system paths
import os


# ======================================================
# DIRECTORY CONFIGURATION
# ======================================================

# Obtain the absolute path of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory containing cleaned CSV files
CSV_CLEAN_DIR = os.path.join(BASE_DIR, "..", "csv_clean")

# Path where the final dataset with engineered features will be stored
DATASET_OUT = os.path.join(BASE_DIR, "..", "dataset_final", "mqtt_ddos_features.csv")

# Create output directory if it does not exist
os.makedirs(os.path.dirname(DATASET_OUT), exist_ok=True)


# ======================================================
# LOAD CLEANED CSV FILES
# ======================================================

# Search for all CSV files inside the cleaned dataset directory
csv_files = glob.glob(os.path.join(CSV_CLEAN_DIR, "*.csv"))

# Stop execution if no files are found
if not csv_files:
    raise RuntimeError("No CSV files found in csv_clean directory")

# List to store processed dataframes
dfs = []


# ======================================================
# FEATURE ENGINEERING PIPELINE
# ======================================================

# Process each cleaned CSV file
for file in csv_files:

    # Extract filename in lowercase for scenario identification
    filename = os.path.basename(file).lower()
    print(f"Processing file: {filename}")

    # Load the cleaned CSV dataset
    df = pd.read_csv(file)

    # --------------------------------------------------
    # SCENARIO LABELING
    # --------------------------------------------------
    # Assign scenario labels based on the filename
    # This allows distinguishing normal traffic from attack traffic

    if "normal" in filename:
        df["scenario"] = "normal"

    elif "normal_ecg" in filename:
        df["scenario"] = "normal_ecg"

    elif "flooding_basic" in filename:
        df["scenario"] = "flooding_basic"

    elif "flooding_delayed" in filename:
        df["scenario"] = "flooding_delayed"

    else:
        df["scenario"] = "unknown"


    # --------------------------------------------------
    # TEMPORAL ORDERING
    # --------------------------------------------------
    # Sort packets chronologically to compute accurate
    # inter-arrival time (IAT) values

    df = df.sort_values("frame.time_relative")


    # --------------------------------------------------
    # FLOW IDENTIFIER
    # --------------------------------------------------
    # Create a unique identifier for each TCP flow

    df["flow_id"] = (
        df["ip.src"].astype(str) + "-" +
        df["ip.dst"].astype(str) + "-" +
        df["tcp.srcport"].astype(str) + "-" +
        df["tcp.dstport"].astype(str)
    )


    # --------------------------------------------------
    # INTER-ARRIVAL TIME (IAT)
    # --------------------------------------------------
    # Compute time difference between consecutive packets
    # belonging to the same network flow

    df["iat"] = df.groupby("flow_id")["frame.time_relative"].diff().fillna(0)


    # --------------------------------------------------
    # TCP FLAG FEATURES
    # --------------------------------------------------
    # Extract binary indicators for key TCP flags commonly
    # associated with flooding attacks

    # Detect SYN packets
    df["is_syn"] = ((df["tcp.flags.dec"] & 0x02) > 0).astype(int)

    # Detect FIN packets
    df["is_fin"] = ((df["tcp.flags.dec"] & 0x01) > 0).astype(int)

    # Detect RST packets
    df["is_rst"] = ((df["tcp.flags.dec"] & 0x04) > 0).astype(int)


    # --------------------------------------------------
    # MQTT PRESENCE FEATURE
    # --------------------------------------------------
    # Binary feature indicating whether the packet contains MQTT

    df["has_mqtt"] = (df["mqtt.msgtype"] != "NO_MQTT").astype(int)


    # --------------------------------------------------
    # TEMPORAL WINDOW (1 SECOND)
    # --------------------------------------------------
    # Aggregate traffic in 1-second windows for rate calculation

    df["time_bin"] = df["frame.time_relative"].astype(int)


    # --------------------------------------------------
    # PACKET RATE
    # --------------------------------------------------
    # Compute the number of packets per flow per second

    pkt_rate = (
        df.groupby(["flow_id", "time_bin"])
        .size()
        .reset_index(name="pkt_rate")
    )

    # Merge packet rate back into the original dataframe
    df = df.merge(pkt_rate, on=["flow_id", "time_bin"], how="left")


    # --------------------------------------------------
    # BYTE RATE
    # --------------------------------------------------
    # Compute the number of transmitted bytes per flow per second

    bytes_rate = (
        df.groupby(["flow_id", "time_bin"])["frame.len"]
        .sum()
        .reset_index(name="bytes_rate")
    )

    # Merge byte rate back into the dataframe
    df = df.merge(bytes_rate, on=["flow_id", "time_bin"], how="left")


    # --------------------------------------------------
    # HANDLE REMAINING MISSING VALUES
    # --------------------------------------------------
    # Replace any remaining NaN values with zero

    df = df.fillna(0)


    # Store processed dataframe
    dfs.append(df)


# ======================================================
# FINAL DATASET GENERATION
# ======================================================

# Combine all scenarios into a single dataset
final_df = pd.concat(dfs, ignore_index=True)

# Save the final dataset with engineered features
final_df.to_csv(DATASET_OUT, index=False)


# Display final dataset statistics
print("\nFEATURE ENGINEERING COMPLETED")
print("Final dataset:", DATASET_OUT)
print("Total rows:", len(final_df))
print("Total columns:", len(final_df.columns))
print("Scenario distribution:")
print(final_df["scenario"].value_counts())