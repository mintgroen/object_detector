# Object Detection with YOLOv8 and MQTT

This script uses a YOLOv8 model to perform object detection on RTSP camera streams. It then publishes the detection results to an MQTT broker, with support for Home Assistant MQTT discovery.

## Features

-   Object detection using YOLOv8 from RTSP camera streams.
-   Publishes detection results to an MQTT broker.
-   Home Assistant MQTT discovery for automatic integration.
-   Creates a single sensor per camera that dynamically displays detected objects.
-   Saves frames with detections to a specified folder.

## Setup

1.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```
    *(Note: a `requirements.txt` file is not yet provided, but should contain `opencv-python`, `ultralytics`, `paho-mqtt`)*

2.  **Configuration:**

    Create a `config.json` file in the `config` directory. See `config/config.json.example` for an example.

    ```json
    {
        "cameras": [
            {
                "name": "camera1",
                "url": "rtsp://user:pass@ip_address/stream",
                "topic": "objectdetection/camera1",
                "output_folder": "/path/to/save/frames"
            }
        ],
        "model_path": "/path/to/your/best.pt",
        "interval": 300,
        "mqtt": {
            "broker": "127.0.0.1",
            "port": 1883,
            "user": null,
            "pass": null
        }
    }
    ```

3.  **Download a YOLOv8 model:**

    Download a pre-trained YOLOv8 model (e.g., `yolov8n.pt`) or train your own model and place it in the specified `model_path`.

## Usage

Run the script:

```bash
python cv_camera.py
```

## Home Assistant Integration

This script uses MQTT discovery to automatically integrate with Home Assistant. For each camera, it creates a single sensor.

-   **Sensor**: A `sensor` is created for each camera (e.g., `sensor.camera1_detections`).
    -   The state of the sensor is a comma-separated string of the names of the detected objects (e.g., "person, car"). If no objects are detected, the state will be "none".
    -   The sensor's attributes contain an array of the detected objects, including their confidence scores.
