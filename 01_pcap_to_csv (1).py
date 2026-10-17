# ======================================================
# PCAP → CSV CONVERSION SCRIPT
# This script automatically converts PCAP files into CSV
# extracting selected network and MQTT features using TShark
# ======================================================

import subprocess
import glob
import os


# ======================================================
# GENERAL CONFIGURATION
# ======================================================

# Get the absolute path of the current script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory containing PCAP files organized by scenario
PCAP_DIR = os.path.join(BASE_DIR, "..", "pcaps")

# Output directory where CSV files will be stored
OUT_DIR = os.path.join(BASE_DIR, "..", "csv_raw")

# Create the output directory if it does not exist
os.makedirs(OUT_DIR, exist_ok=True)

# Path to the TShark executable (Wireshark command-line tool)
TSHARK = r"C:\Program Files\Wireshark\tshark.exe"


# Print configuration information for debugging and validation
print("========================================")
print("PCAP_DIR:", PCAP_DIR)
print("Exists PCAP_DIR:", os.path.exists(PCAP_DIR))
print("TSHARK:", TSHARK)
print("Exists TSHARK:", os.path.exists(TSHARK))
print("========================================")


# Verify that the PCAP directory exists
if not os.path.exists(PCAP_DIR):
    raise FileNotFoundError("PCAP directory not found")

# Verify that TShark executable exists
if not os.path.exists(TSHARK):
    raise FileNotFoundError("tshark.exe not found")


# ======================================================
# NETWORK AND MQTT FEATURES TO EXTRACT
# ======================================================

# List of packet fields extracted from PCAP using TShark
# These features are later used for traffic analysis and ML models
FIELDS = [
    "frame.time_relative",       # Packet timestamp relative to capture start
    "ip.src",                    # Source IP address
    "ip.dst",                    # Destination IP address
    "tcp.srcport",               # TCP source port
    "tcp.dstport",               # TCP destination port
    "tcp.flags",                 # TCP flag field (SYN, ACK, etc.)
    "tcp.window_size_value",     # TCP window size
    "tcp.analysis.ack_rtt",      # Round-trip time estimation
    "frame.len",                 # Packet length in bytes
    "mqtt.msgtype",              # MQTT message type
    "mqtt.qos",                  # MQTT quality of service level
    "mqtt.clientid",             # MQTT client identifier
    "mqtt.topic"                 # MQTT topic name
]


# ======================================================
# PCAP → CSV PROCESSING PIPELINE
# ======================================================

# Iterate through each scenario folder inside the PCAP directory
for scenario in os.listdir(PCAP_DIR):

    scenario_path = os.path.join(PCAP_DIR, scenario)

    # Skip files that are not directories
    if not os.path.isdir(scenario_path):
        continue


    # --------------------------------------------------
    # SCENARIO NAME NORMALIZATION
    # This section standardizes scenario labels
    # so they can be used consistently for ML training
    # --------------------------------------------------

    if "normal" in scenario.lower():
        scenario_name = "normal"

    elif "normal_ecg" in scenario.lower():
        scenario_name = "normal_ecg"

    elif "flooding_basic" in scenario.lower():
        scenario_name = "flooding_basic"

    elif "flooding_delayed" in scenario.lower():
        scenario_name = "flooding_delayed"

    else:
        scenario_name = scenario.lower()


    print(f"\nProcessing scenario: {scenario} → labeled as: {scenario_name}")


    # Search for all PCAP files inside the scenario directory
    pcap_files = glob.glob(os.path.join(scenario_path, "*.pcap"))

    if not pcap_files:
        print("  No PCAP files found in this scenario")
        continue


    # --------------------------------------------------
    # PROCESS EACH PCAP FILE
    # --------------------------------------------------

    for pcap in pcap_files:

        # Generate CSV output filename
        name = os.path.basename(pcap).replace(".pcap", ".csv")
        out_csv = os.path.join(OUT_DIR, f"{scenario_name}_{name}")


        # Build the TShark command
        cmd = [
            TSHARK,

            "-r", pcap,            # Read packets from the input PCAP file

            "-Y", "tcp",           # Apply display filter (only TCP packets)

            "-T", "fields",        # Output format: fields (structured extraction)

            "-E", "header=y",      # Include column headers

            "-E", "separator=,",   # Use comma as CSV separator

            "-E", "quote=d",       # Quote fields with double quotes
        ]


        # Add each requested field to the extraction command
        for f in FIELDS:
            cmd += ["-e", f]


        # --------------------------------------------------
        # EXECUTE TSHARK EXTRACTION
        # --------------------------------------------------

        try:

            # Open the output CSV file
            with open(out_csv, "w", encoding="utf-8") as f:

                # Run TShark and redirect stdout to the CSV file
                subprocess.run(cmd, stdout=f, stderr=subprocess.DEVNULL)

            print(f"CSV generated: {out_csv}")

        except Exception as e:

            # Handle errors during PCAP processing
            print(f"Error processing {pcap}")
            print(e)


print("\nPROCESS COMPLETED")