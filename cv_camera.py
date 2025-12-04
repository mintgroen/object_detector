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


def publish_mqtt_discovery(client, cameras, model_names):
    """Publishes MQTT discovery messages for Home Assistant for each object class."""
    for camera in cameras:
        camera_name = camera["name"]
        for class_name in model_names:
            safe_class_name = class_name.replace(" ", "_")
            discovery_topic = f"homeassistant/binary_sensor/object_detection/{camera_name}_{safe_class_name}/config"
            
            state_topic = f"objectdetection/{camera_name}/{safe_class_name}/state"
            attributes_topic = f"objectdetection/{camera_name}/{safe_class_name}/attributes"

            payload = {
                "name": f"{camera_name} {class_name}",
                "state_topic": state_topic,
                "json_attributes_topic": attributes_topic,
                "payload_on": "ON",
                "payload_off": "OFF",
                "device_class": "presence",
                "unique_id": f"cv_camera_{camera_name}_{safe_class_name}",
                "device": {
                    "identifiers": [f"cv_camera_{camera_name}"],
                    "name": f"Object Detection Camera - {camera_name}",
                    "model": "YOLOv8 Object Detection",
                    "manufacturer": "Custom"
                }
            }
            client.publish(discovery_topic, json.dumps(payload), retain=True)
            logging.info(f"Published MQTT discovery for {camera_name} {class_name} sensor.")




def main():
    # 1. Load the Model once at startup
    logging.info(f"Loading model from {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
        class_names = model.names
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
        publish_mqtt_discovery(client, CAMERAS, class_names)

    except Exception as e:
        logging.error(f"MQTT Connection failed: {e}")
        return

    # 3. Main Loop
    logging.info(f"Starting loop. Capturing every {INTERVAL} seconds.")
    
    previous_detections = {camera['name']: set() for camera in CAMERAS}

    while True:
        for camera in CAMERAS:
            start_time = time.time()
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

                # --- D. Publish to MQTT ---
                detected_classes = {d['object'] for d in detections}

                # Publish ON for newly detected objects
                for d in detections:
                    class_name = d['object']
                    safe_class_name = class_name.replace(" ", "_")
                    confidence_payload = json.dumps({'confidence': d['confidence']})
                    
                    attributes_topic = f"objectdetection/{camera_name}/{safe_class_name}/attributes"
                    state_topic = f"objectdetection/{camera_name}/{safe_class_name}/state"
                    
                    client.publish(attributes_topic, confidence_payload)
                    client.publish(state_topic, "ON")
                    logging.info(f"Published ON for {camera_name} - {class_name}")

                # Publish OFF for objects that are no longer detected
                disappeared_classes = previous_detections[camera_name] - detected_classes
                for class_name in disappeared_classes:
                    safe_class_name = class_name.replace(" ", "_")
                    state_topic = f"objectdetection/{camera_name}/{safe_class_name}/state"
                    client.publish(state_topic, "OFF")
                    logging.info(f"Published OFF for {camera_name} - {class_name}")

                # Update previous detections for the next iteration
                previous_detections[camera_name] = detected_classes

            else:
                logging.warning(f"Skipping prediction for {camera_name} due to camera error.")

            # --- E. Sleep ---
            # Calculate execution time to keep the interval accurate
            elapsed = time.time() - start_time
            sleep_time = max(0, INTERVAL - elapsed)
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()