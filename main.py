from flask import Flask, jsonify, Response
from flask_socketio import SocketIO
import cv2
import os
import json
import atexit

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

data_dir = "mock_data"  # directory for mock files
privacy_mode = False
patient_privacy_mode = False

# Webcam init with explicit Linux backend (CAP_V4L2)
camera = cv2.VideoCapture(0, cv2.CAP_V4L2)

def tail(filepath, n=10):
    with open(filepath, 'r') as f:
        return f.readlines()[-n:]

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

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

@app.route('/get_log_data')
def get_log_data():
    file_path = os.path.join(data_dir, 'daily_log.txt')
    if os.path.exists(file_path):
        lines = tail(file_path)
        return jsonify({"logs": [line.strip() for line in lines]})
    return jsonify({"logs": []})

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_daily_logs')
def get_daily_logs():
    file_path = os.path.join(data_dir, 'logs.json')
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return jsonify(json.load(f)[-5:])
    return jsonify([])

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

# Release camera cleanly on app exit
@atexit.register
def cleanup():
    if camera.isOpened():
        camera.release()

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
