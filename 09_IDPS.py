# =====================================================
# PRODUCTION IDPS – NEURO-FUZZY DETECTION ENGINE
# Real-time intrusion detection and prevention system
# for MQTT traffic using a hybrid neuro-fuzzy model.
# =====================================================

import subprocess
import joblib
import numpy as np
from collections import defaultdict
import ipaddress
import time

# =====================================================
# CONFIGURATION
# Set the public IP address of the monitored server
# =====================================================

SERVER_PUBLIC_IP = "__.__.__.___"
MODEL_PATH = "models/neuro_fuzzy_tree_model.pkl"
MQTT_PORT = "1883"

# Number of consecutive detections required to trigger blocking
ATTACK_REPEAT_LIMIT = 10

# Heuristic packet rate threshold used for flooding detection
FLOOD_PKT_RATE = 200

# Whitelisted IP addresses that must never be blocked
WHITELIST = {
    SERVER_PUBLIC_IP,
    "127.0.0.1",
    "172.18.0.1",
    "172.18.0.31"
}

print("Server public IP:", SERVER_PUBLIC_IP)
print("IPS engine started\n")

# =====================================================
# NETWORK UTILITY FUNCTIONS
# =====================================================

def is_public(ip):
    """
    Check whether an IP address is globally routable.
    """
    try:
        return ipaddress.ip_address(ip).is_global
    except:
        return False


def should_analyze(src, dst):
    """
    Determine whether the packet should be analyzed.
    Only public external sources targeting the monitored server are evaluated.
    """

    if src in WHITELIST:
        return False

    if dst != SERVER_PUBLIC_IP:
        return False

    if not is_public(src):
        return False

    return True


def has_active_connections(ip):
    """
    Check whether active connections from the IP exist
    in the connection tracking table.
    """
    result = subprocess.getoutput(f"conntrack -L | grep {ip}")
    return len(result.strip()) > 0


def block_ip(ip):
    """
    Block an attacker IP using iptables and
    remove active connections using conntrack.
    """

    if ip in WHITELIST:
        print("Attempted block of protected IP:", ip)
        return

    print("\nBlocking attacker IP:", ip)

    subprocess.run([
        "iptables",
        "-I", "DOCKER-USER", "1",
        "-s", ip,
        "-j", "DROP"
    ])

    subprocess.run(
        ["conntrack", "-D", "-s", ip],
        stderr=subprocess.DEVNULL
    )

# =====================================================
# LOAD TRAINED MODEL
# =====================================================

bundle = joblib.load(MODEL_PATH)

trees = bundle["trees"]
feature_subsets = bundle["feature_subsets"]
mlp = bundle["mlp"]
scaler = bundle["scaler"]
threshold = bundle["threshold"]
k_sharpness = bundle["k_sharpness"]

print("Decision threshold:", round(threshold,4))
print("Sigmoid sharpness:", k_sharpness)
print("Detection repetitions required:", ATTACK_REPEAT_LIMIT)
print("Monitoring external MQTT traffic\n")

# =====================================================
# FUZZY MEMBERSHIP FUNCTIONS
# =====================================================

def sigmoid_membership(x, threshold, sharpness):
    """
    Sigmoid membership function used to smooth
    decision boundaries in the fuzzy tree evaluation.
    """

    z = sharpness * (x - threshold)

    # Prevent numerical overflow
    z = np.clip(z, -60, 60)

    return 1 / (1 + np.exp(-z))


def fuzzy_tree_output(tree, x, selected_features):
    """
    Evaluate a fuzzy decision tree using soft thresholds.
    """

    node = 0
    x_sub = x[selected_features]

    while tree.tree_.children_left[node] != -1:

        feature = tree.tree_.feature[node]
        threshold_val = tree.tree_.threshold[node]

        mu = sigmoid_membership(
            x_sub[feature],
            threshold_val,
            k_sharpness
        )

        left = tree.tree_.children_left[node]
        right = tree.tree_.children_right[node]

        left_prob = tree.tree_.value[left][0][1]
        right_prob = tree.tree_.value[right][0][1]

        # Fuzzy interpolation between child nodes
        prob = mu * right_prob + (1 - mu) * left_prob

        node = right if mu > 0.5 else left

    return prob


def generate_tree_features(X_input):
    """
    Generate meta-features using the ensemble of fuzzy trees.
    These features are later processed by the neural meta-classifier.
    """

    outputs = []

    for x in X_input:

        tree_scores = []

        for i, tree in enumerate(trees):

            score = fuzzy_tree_output(
                tree,
                x,
                feature_subsets[i]
            )

            tree_scores.append(score)

        outputs.append(tree_scores)

    return np.array(outputs)

# =====================================================
# FLOW STATE TRACKING (PER SOURCE IP)
# =====================================================

flows = defaultdict(lambda: {
    "pkt_count": 0,
    "byte_sum": 0,
    "iat_values": [],
    "last_time": None,
    "time_bin": None
})

attack_registry = {}
attack_start_time = {}

# =====================================================
# MQTT TRAFFIC CAPTURE (TSHARK)
# =====================================================

cmd = [
    "tshark",
    "-i", "any",
    "-f", f"tcp dst port {MQTT_PORT} and dst host {SERVER_PUBLIC_IP}",
    "-T", "fields",
    "-e", "frame.time_relative",
    "-e", "ip.src",
    "-e", "ip.dst",
    "-e", "tcp.srcport",
    "-e", "tcp.dstport",
    "-e", "frame.len",
    "-l"
]

proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)

# =====================================================
# MAIN PROCESSING LOOP
# =====================================================

for line in proc.stdout:

    parts = line.strip().split("\t")

    if len(parts) != 6:
        continue

    try:

        t = float(parts[0])
        src = parts[1]
        dst = parts[2]
        sport = parts[3]
        dport = parts[4]
        flen = int(parts[5])

    except:
        continue

    if not should_analyze(src, dst):
        continue

    if src in attack_registry and attack_registry[src]["blocked"]:
        continue

    flow_id = src
    time_bin = int(t)

    f = flows[flow_id]

    # =====================================================
    # NEW TIME WINDOW (1 SECOND)
    # =====================================================

    if f["time_bin"] is not None and time_bin != f["time_bin"]:

        if f["pkt_count"] > 0:

            iat = np.mean(f["iat_values"]) if f["iat_values"] else 0
            pkt_rate = f["pkt_count"]
            bytes_rate = f["byte_sum"]
            frame_len = f["byte_sum"] / f["pkt_count"]

            # =====================================================
            # FLOODING HEURISTIC
            # =====================================================

            if pkt_rate > FLOOD_PKT_RATE:

                score = 1.0

            else:

                X = np.array([[iat, pkt_rate, bytes_rate, frame_len]])
                Xs = scaler.transform(X)

                tree_features = generate_tree_features(Xs)
                score = mlp.predict_proba(tree_features)[0,1]

            print("FLOW:", flow_id, "score:", round(score,4))

            # =====================================================
            # ATTACK DETECTION LOGIC
            # =====================================================

            if score >= threshold:

                if src not in attack_registry:

                    attack_registry[src] = {
                        "count": 1,
                        "blocked": False
                    }

                    attack_start_time[src] = time.time()

                else:

                    attack_registry[src]["count"] += 1

                print(
                    "Attack detected from",
                    src,
                    attack_registry[src]["count"],
                    "/",
                    ATTACK_REPEAT_LIMIT
                )

                if attack_registry[src]["count"] >= ATTACK_REPEAT_LIMIT:

                    if has_active_connections(src):

                        detection_time = time.time() - attack_start_time[src]

                        print(
                            "Detection time:",
                            round(detection_time,3),
                            "seconds"
                        )

                        block_ip(src)

                        attack_registry[src]["blocked"] = True

            else:

                # Counter decay mechanism (does not reset completely)
                if src in attack_registry:

                    attack_registry[src]["count"] = max(
                        attack_registry[src]["count"] - 1,
                        0
                    )

        # Reset flow statistics

        f["pkt_count"] = 0
        f["byte_sum"] = 0
        f["iat_values"] = []

    # =====================================================
    # UPDATE FLOW STATISTICS
    # =====================================================

    if f["last_time"] is not None:
        f["iat_values"].append(t - f["last_time"])

    f["last_time"] = t
    f["pkt_count"] += 1
    f["byte_sum"] += flen
    f["time_bin"] = time_bin