# cv_camera1.py
import cv2
import time
import json
import logging
from ultralytics import YOLO
import paho.mqtt.client as mqtt
from datetime import datetime
import os

# --- Configuration Loading ---
def load_config():
    with open("config/config.json", "r") as f:
        return json.load(f)

config = load_config()

CAMERAS = config["cameras"]
MODEL_PATH = config["model_path"]
INTERVAL = config["interval"]
MQTT_BROKER = config["mqtt"]["broker"]
MQTT_PORT = config["mqtt"]["port"]
MQTT_USER = config["mqtt"]["user"]
MQTT_PASS = config["mqtt"]["pass"]

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_camera_frame(rtsp_url):
    """Connects to RTSP, clears buffer, captures one frame, and disconnects."""
    # Force TCP for stability
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        logging.error(f"Could not open RTSP stream at {rtsp_url}")
        return None

    # Read a few frames to clear buffer/auto-exposure (warmup)
    for _ in range(5):
        cap.read()

    ret, frame = cap.read()
    cap.release() # Release immediately to save resources during sleep

    if ret:
        return frame
    else:
        logging.error("Failed to retrieve frame")
        return None

def save_frame(frame, folder, camera_name):
    """Saves the frame to the specified folder."""
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except OSError as e:
            logging.error(f"Failed to create directory {folder}: {e}")
            return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{folder}/{camera_name}_{timestamp}.jpg"
    try:
        cv2.imwrite(filename, frame)
        logging.info(f"Saved frame to {filename}")
    except Exception as e:
        logging.error(f"Failed to save frame {filename}: {e}")


def publish_mqtt_discovery(client, cameras):
    """Publishes MQTT discovery messages for Home Assistant."""
    for camera in cameras:
        camera_name = camera["name"]
        topic = camera["topic"]
        
        # Discovery for binary sensor (object detection)
        discovery_topic = f"homeassistant/binary_sensor/object_detection/{camera_name}/config"
        payload = {
            "name": f"{camera_name} Object Detected",
            "state_topic": topic,
            "value_template": "{% if value_json.count > 0 %}ON{% else %}OFF{% endif %}",
            "device_class": "motion",
            "unique_id": f"cv_camera_{camera_name}_motion",
            "json_attributes_topic": topic,
            "device": {
                "identifiers": [f"cv_camera_{camera_name}"],
                "name": f"Object Detection Camera - {camera_name}",
                "model": "YOLOv8 Object Detection",
                "manufacturer": "Custom"
            }
        }
        client.publish(discovery_topic, json.dumps(payload), retain=True)
        logging.info(f"Published MQTT discovery for {camera_name} motion sensor.")

        # Discovery for sensor (detection count)
        discovery_topic_sensor = f"homeassistant/sensor/object_detection/{camera_name}_count/config"
        payload_sensor = {
            "name": f"{camera_name} Detection Count",
            "state_topic": topic,
            "value_template": "{{ value_json.count }}",
            "unique_id": f"cv_camera_{camera_name}_count",
            "json_attributes_topic": topic,
            "device": {
                "identifiers": [f"cv_camera_{camera_name}"],
            }
        }
        client.publish(discovery_topic_sensor, json.dumps(payload_sensor), retain=True)
        logging.info(f"Published MQTT discovery for {camera_name} detection count sensor.")



def main():
    # 1. Load the Model once at startup
    logging.info(f"Loading model from {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return

    # 2. Setup MQTT Client
    client = mqtt.Client()
    if MQTT_USER and MQTT_PASS:
        client.username_pw_set(MQTT_USER, MQTT_PASS)

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start() # Start network loop in background
        logging.info("Connected to MQTT Broker")
        
        # Publish discovery messages
        publish_mqtt_discovery(client, CAMERAS)

    except Exception as e:
        logging.error(f"MQTT Connection failed: {e}")
        return

    # 3. Main Loop
    logging.info(f"Starting loop. Capturing every {INTERVAL} seconds.")

    while True:
        for camera in CAMERAS:
            start_time = time.time()

            # --- A. Capture ---
            frame = get_camera_frame(camera["url"])

            if frame is not None:
                # --- Save Frame ---
                if "output_folder" in camera:
                    save_frame(frame, camera["output_folder"], camera["name"])

                # --- B. Predict ---
                # 'conf=0.5' means only detect objects with >50% confidence
                results = model.predict(frame, conf=0.5, verbose=False)

                # --- C. Process Results ---
                detections = []

                # Iterate through detections in the first (and only) frame
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        class_name = model.names[cls_id]
                        confidence = float(box.conf[0])

                        detections.append({
                            "object": class_name,
                            "confidence": round(confidence, 2)
                        })

                # --- D. Publish to MQTT ---
                payload = {
                    "timestamp": datetime.now().isoformat(),
                    "camera": camera["name"],
                    "count": len(detections),
                    "detections": detections
                }

                json_payload = json.dumps(payload)
                client.publish(camera["topic"], json_payload)
                logging.info(f"Published to {camera['topic']}: {json_payload}")

            else:
                logging.warning(f"Skipping prediction for {camera['name']} due to camera error.")

            # --- E. Sleep ---
            # Calculate execution time to keep the 5-minute schedule accurate
            elapsed = time.time() - start_time
            sleep_time = max(0, INTERVAL - elapsed)
            time.sleep(sleep_time)

if __name__ == "__main__":
    main()