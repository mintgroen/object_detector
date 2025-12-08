# cv_camera1.py
import os
import cv2
import time
import json
import logging
from ultralytics import YOLO
import paho.mqtt.client as mqtt
from datetime import datetime


# --- Configuration Loading ---
def load_config():
    with open("config/config.json", "r") as f:
        return json.load(f)

config = load_config()
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

CAMERAS = config["cameras"]
MODEL_PATH = config["model_path"]
INTERVAL = config["interval"]
MQTT_BROKER = config["mqtt"]["broker"]
MQTT_PORT = config["mqtt"]["port"]
MQTT_USER = config["mqtt"]["user"]
MQTT_PASS = config["mqtt"]["pass"]

# Logging Setup
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def get_camera_frame(rtsp_url):
    """Connects to RTSP, clears buffer, captures one frame, and disconnects."""
    logging.debug(f"Connecting to RTSP stream: {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        logging.error(f"Could not open RTSP stream at {rtsp_url}")
        return None

    # Read a few frames to clear buffer/auto-exposure (warmup)
    for _ in range(5):
        ret, _ = cap.read()
        if not ret:
            logging.warning(f"Failed to read frame during warmup for {rtsp_url}")
            break

    ret, frame = cap.read()
    cap.release() # Release immediately to save resources during sleep

    if ret:
        return frame
    else:
        logging.error("Failed to retrieve frame")
        return None

def save_frame(frame, folder, camera_name):
    """Saves the frame to the specified folder."""
    logging.debug(f"Attempting to save frame for {camera_name} in {folder}")
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
        
        state_topic = f"objectdetection/{camera_name}/state"
        attributes_topic = f"objectdetection/{camera_name}/attributes"

        discovery_topic = f"homeassistant/sensor/object_detection/{camera_name}/config"
        payload = {
            "name": f"{camera_name} Detections",
            "state_topic": state_topic,
            "value_template": "{{ value_json.detections | map(attribute='object') | join(', ') }}",
            "json_attributes_topic": state_topic,
            "json_attributes_template": "{{ value_json | tojson }}",
            "unique_id": f"cv_camera_{camera_name}_detections",
            "device": {
                "identifiers": [f"cv_camera_{camera_name}"],
                "name": f"Object Detection Camera - {camera_name}",
                "model": "YOLOv8 Object Detection",
                "manufacturer": "Custom"
            }
        }
        client.publish(discovery_topic, json.dumps(payload), retain=True)
        logging.debug(f"Published discovery payload for {camera_name}: {json.dumps(payload)}")
        logging.info(f"Published MQTT discovery for {camera_name} detections sensor.")




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
        start_time = time.time() # Start time for the whole iteration

        for camera in CAMERAS:
            camera_name = camera["name"]

            # --- A. Capture ---
            frame = get_camera_frame(camera["url"])

            if frame is not None:
                # --- Save Frame ---
                if "output_folder" in camera:
                    save_frame(frame, camera["output_folder"], camera_name)

                # --- B. Predict ---
                results = model.predict(frame, conf=0.5, verbose=False)

                # --- C. Process Results ---
                detections = []
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        class_name = model.names[cls_id]
                        confidence = float(box.conf[0])
                        detections.append({
                            "object": class_name,
                            "confidence": round(confidence, 2)
                        })

                logging.debug(f"Raw detections for {camera_name}: {detections}")

                # --- D. Publish to MQTT ---
                state_topic = f"objectdetection/{camera_name}/state"

                if detections:
                    payload = {
                        "detections": detections,
                        "count": len(detections)
                    }
                else:
                    payload = {
                        "detections": [{"object": "none", "confidence": 1.0}],
                        "count": 0
                    }

                payload_str = json.dumps(payload)
                logging.debug(f"Publishing to {state_topic}: {payload_str}")
                client.publish(state_topic, payload_str)
                
                logging.info(f"Published detections for {camera_name}: {payload['detections']}")

            else:
                logging.warning(f"Skipping prediction for {camera_name} due to camera error.")

        # --- E. Sleep ---
        # Calculate execution time for all cameras
        elapsed = time.time() - start_time
        sleep_time = max(0, INTERVAL - elapsed)
        
        logging.info(f"All cameras processed. Sleeping for {sleep_time:.2f} seconds.")
        time.sleep(sleep_time)


if __name__ == "__main__":
    main()