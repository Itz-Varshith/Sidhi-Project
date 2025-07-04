# from flask import Flask, jsonify, Response, render_template
# from flask_socketio import SocketIO
# import cv2
# import os
# import json
# import atexit
# import threading
# import queue
# from flask_cors import CORS
# from ultralytics import YOLO
# import subprocess




# app = Flask(__name__)
# CORS(app)
# socketio = SocketIO(app, cors_allowed_origins="*")

# data_dir = "mock_data"  # directory for mock files
# privacy_mode = False
# patient_privacy_mode = False

# # Webcam init with explicit Linux backend (CAP_V4L2)
# camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
# camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# # Thread-safe queue for frames
# frame_queue = queue.Queue(maxsize=1)
# stop_event = threading.Event()

# model_path = "backend/weights/incrementalv8.pt" 
# model = YOLO(model_path)  # Automatically uses CUDA if available

# # Get class names from the model
# CLASS_NAMES = model.names  # This is a dict: {0: 'class0', 1: 'class1', ...}

# def camera_detection_thread():
#     import numpy as np
#     while not stop_event.is_set():
#         success, new_frame = camera.read()
#         if not success:
#             continue
#         # YOLOv8 expects BGR images (OpenCV default)
#         results = model(new_frame, stream=True)
#         for result in results:
#             boxes = result.boxes
#             for box in boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
#                 confidence = float(box.conf[0])
#                 class_id = int(box.cls[0])
#                 label = CLASS_NAMES.get(class_id, f"Class {class_id}")
#                 cv2.rectangle(new_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                 cv2.putText(new_frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#         ret, buffer = cv2.imencode('.jpg', new_frame)
#         if ret:
#             # Always keep only the latest frame
#             if not frame_queue.empty():
#                 try:
#                     frame_queue.get_nowait()
#                 except queue.Empty:
#                     pass
#             frame_queue.put(buffer.tobytes())

# # Start the detection thread
# detection_thread = threading.Thread(target=camera_detection_thread, daemon=True)
# detection_thread.start()

# @socketio.on("toggle_privacy")
# def toggle_privacy(data):
#     global privacy_mode
#     privacy_mode = data["enabled"]
#     print(f"Privacy mode enabled: {privacy_mode}")

# @socketio.on("toggle_patient_privacy")
# def toggle_patient_privacy(data):
#     global patient_privacy_mode
#     patient_privacy_mode = data["enabled"]
#     print(f"Patient Privacy mode enabled: {patient_privacy_mode}")

# @app.route('/')
# def index():
#     return render_template('AIIMS.html')

# @app.route('/video')
# def video():
#     def generate():
#         while True:
#             try:
#                 frame = frame_queue.get(timeout=1)
#                 yield (b'--frame\r\n'
#                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#             except queue.Empty:
#                 continue
#     return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/get_log_data')
# def get_log_data():
#     file_path = os.path.join(data_dir, 'daily_logs.json')
#     if os.path.exists(file_path):
#         with open(file_path, 'r') as f:
#             logs = json.load(f)
#         return jsonify(logs)
#     return jsonify([])

# @app.route('/get_daily_logs')
# def get_daily_logs():
#     file_path = os.path.join(data_dir, 'logs.json')
#     if os.path.exists(file_path):
#         with open(file_path, 'r') as f:
#             return jsonify(json.load(f))
#     return jsonify([])

# @app.route('/start_training', methods=['POST'])
# def start_training():
#     # Adjust the path to auto_train.py as needed
#     subprocess.Popen(["python", "auto_train.py"])
#     return jsonify({"status": "Training started"})

# @app.route('/get_ventilator_data')
# def get_ventilator_data():
#     file_path = os.path.join(data_dir, 'ventilator.json')
#     if os.path.exists(file_path):
#         with open(file_path, 'r') as f:
#             return jsonify(json.load(f)[-10:])
#     return jsonify([])

# @app.route('/get_paramonitor_data')
# def get_paramonitor_data():
#     file_path = os.path.join(data_dir, 'paramonitor.json')
#     if os.path.exists(file_path):
#         with open(file_path, 'r') as f:
#             return jsonify(json.load(f)[-10:])
#     return jsonify([])

# @atexit.register
# def cleanup():
#     stop_event.set()
#     detection_thread.join(timeout=2)
#     if camera.isOpened():
#         camera.release()

# if __name__ == '__main__':
#     socketio.run(app, host='0.0.0.0', port=3000, debug=False)




from flask import Flask, jsonify, Response, render_template
from flask_socketio import SocketIO
import cv2
import os
import json
import atexit
import threading
import queue
from flask_cors import CORS
from ultralytics import YOLO
import subprocess

from auto_train import run_autotrain

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

data_dir = "mock_data"  # directory for mock files
privacy_mode = False
patient_privacy_mode = False

# Webcam init with explicit Linux backend (CAP_V4L2)
camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Thread-safe queue for frames
frame_queue = queue.Queue(maxsize=1)
stop_event = threading.Event()

model_path = "backend/weights/incrementalv8.pt"
model = YOLO(model_path)  # Automatically uses CUDA if available

# Get class names from the model
CLASS_NAMES = model.names  # This is a dict: {0: 'class0', 1: 'class1', ...}

def camera_detection_thread():
    import numpy as np
    while not stop_event.is_set():
        success, new_frame = camera.read()
        if not success:
            continue
        # Set confidence and IoU thresholds to reduce duplicate boxes
        results = model(new_frame, stream=True, conf=0.5, iou=0.4)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                label = CLASS_NAMES.get(class_id, f"Class {class_id}")
                cv2.rectangle(new_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(new_frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', new_frame)
        if ret:
            # Always keep only the latest frame
            if not frame_queue.empty():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            frame_queue.put(buffer.tobytes())

# Start the detection thread
detection_thread = threading.Thread(target=camera_detection_thread, daemon=True)
detection_thread.start()

@socketio.on("toggle_privacy")
def toggle_privacy(data):
    global privacy_mode
    privacy_mode = data["enabled"]
    print(f"Privacy mode enabled: {privacy_mode}")

@socketio.on("toggle_patient_privacy")
def toggle_patient_privacy(data):
    global patient_privacy_mode
    patient_privacy_mode = data["enabled"]
    print(f"Patient Privacy mode enabled: {patient_privacy_mode}")

@app.route('/')
def index():
    return render_template('AIIMS.html')

@app.route('/video')
def video():
    def generate():
        while True:
            try:
                frame = frame_queue.get(timeout=1)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except queue.Empty:
                continue
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_log_data')
def get_log_data():
    file_path = os.path.join(data_dir, 'daily_logs.json')
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            logs = json.load(f)
        return jsonify(logs)
    return jsonify([])

@app.route('/get_daily_logs')
def get_daily_logs():
    file_path = os.path.join(data_dir, 'logs.json')
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return jsonify(json.load(f))
    return jsonify([])

@app.route('/start_training', methods=['POST'])
def start_training():
    subprocess.Popen(["python", "auto_train.py"])
    return jsonify({"status": "Training started"})

@app.route('/get_ventilator_data')
def get_ventilator_data():
    file_path = os.path.join(data_dir, 'ventilator.json')
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return jsonify(json.load(f)[-10:])
    return jsonify([])

@app.route('/get_paramonitor_data')
def get_paramonitor_data():
    file_path = os.path.join(data_dir, 'paramonitor.json')
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return jsonify(json.load(f)[-10:])
    return jsonify([])

@app.route('/autotrain', methods=['GET'])
def train_route():
    result = run_autotrain()
    return jsonify(result)


@atexit.register
def cleanup():
    stop_event.set()
    detection_thread.join(timeout=2)
    if camera.isOpened():
        camera.release()

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=3000, debug=False)