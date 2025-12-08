# Object Detection with YOLOv8 and MQTT

This script uses a YOLOv8 model to perform object detection on RTSP camera streams. It then publishes the detection results to an MQTT broker, with support for Home Assistant MQTT discovery.

## Features

-   Object detection using YOLOv8 from RTSP camera streams.
-   Publishes detection results to an MQTT broker.
-   Home Assistant MQTT discovery for automatic integration.
-   Creates a single sensor per camera that dynamically displays detected objects.
-   Saves frames with detections to a specified folder.

## Home Assistant Integration

This script uses MQTT discovery to automatically integrate with Home Assistant. For each camera, it creates a single sensor that publishes its state and attributes in a single JSON payload.

-   **Sensor**: A `sensor` is created for each camera (e.g., `sensor.camera1_detections`).
    -   The state of the sensor is a comma-separated string of the names of the detected objects (e.g., "person, car"). If no objects are detected, the state will be "none".
    -   The sensor's attributes contain the full JSON payload, which includes a `detections` array (with `object` and `confidence` for each detection) and a `count` of the detected objects.

## Setup

1.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Configuration:

    Create a `config.json` file in the `config` directory.

    ```json
    {
        "cameras": [
            {
                "name": "front_door",
                "url": "rtsp://user:pass@192.168.1.10/stream1",
                "output_folder": "captures/front_door"
            }
        ],
        "model_path": "yolov8n.pt",
        "interval": 10,
        "mqtt": {
            "broker": "192.168.1.100",
            "port": 1883,
            "user": "mqtt_user",
            "pass": "mqtt_password"
        }
    }
    ```

3.  **Download a YOLOv8 model:**

    Download a pre-trained YOLOv8 model (e.g., `yolov8n.pt`) and place it in the project directory, or specify a full path in `config.json`.

## Usage

Run the script:

```bash
python cv_camera.py
```
