'''This project is focused on object detection and event tracking.
Now, when I wrote this code only god and I knew what it does or
how it does, now only god knows. I am not a religious
but this code forces me to look at him for help.'''
import threading
from gc import enable
from ultralytics import YOLO
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
from collections import Counter
import numpy as np
import cv2
import torch
import queue
import pandas as pd
import re
import sys
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, HubertModel
from PIL import Image
import time
import logging
from datetime import datetime, timedelta
import os
from onvif import ONVIFCamera
import shutil
import glob
from multiprocessing import Process

excel_output_dir = "excel_output"

os.makedirs(excel_output_dir, exist_ok=True)
# image_output_dir = 'captured_images'
# if not os.path.exists(image_output_dir):
#     os.makedirs(image_output_dir)
# else:
#     for file in os.listdir(image_output_dir):
#         if file:
#             path = os.path.join(image_output_dir, file)
#             os.remove(path)


app = Flask(__name__)
socketio = SocketIO(app)
privacy_mode = False  # Initial state of privacy mode
patient_privacy_mode  = False
log = logging.getLogger('werkzeug')
log.disabled = True
app.logger.disabled = True
action = 'cancel'
monitor_till = 0
bed_no = 0
bed_to_pan = {1: -.554, 2: -.149, 3:.16, 4:.559}

# Event handler for toggling privacy mode
@socketio.on("toggle_privacy")
def toggle_privacy(data):
    global privacy_mode
    privacy_mode = data["enabled"]
    print(f"Privacy mode enabled: {privacy_mode}")
@socketio.on("toggle_patient_privacy")
def toggle_privacy(data):
    global patient_privacy_mode
    patient_privacy_mode = data["enabled"]
    print(f"Patient Privacy mode enabled: {patient_privacy_mode}")

# camera = cv2.VideoCapture('rtsp://10.2.37.16:554/live/av0')
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

frame_queue = queue.Queue(maxsize=5)

# Global variables
application = ''
start_time = ''
end_time = ''
monitor_time = 8
# Load YOLOv7 model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model_path = r'C:\Users\OMEN\PycharmProjects\object_detection\yolov7\runs\object_detection_model\exp4\weights\best.pt'
# model = torch.hub.load("WongKinYiu/yolov7", "custom", model_path, trust_repo=True)
# print("************************************ :", next(model.parameters()).is_cuda)
camera_state = {'pan': 0, 'tilt': 0}
labels = {0: 'Paramonitor', 1: 'Ventilator'}


def move_and_zoom_camera(ptz_service, profile, pan, tilt, zoom=0.0, speed=.8):
    try:
        ptz_service.AbsoluteMove({
            'ProfileToken': profile.token,
            'Position': {
                'PanTilt': {'x': pan, 'y': tilt},
                'Zoom': {'x': zoom}
            },
            'Speed': {'PanTilt': {'x': speed, 'y': speed}, 'Zoom': speed}
        })
        # print(f"Camera moved to Pan: {pan}, Tilt: {tilt}, Zoom: {zoom}")
    except Exception as e:
        print(f"Error moving PTZ camera: {e}")


def calculate_new_pan_tilt_and_zoom(object_center, object_bbox, frame_dimensions, current_pan, current_tilt,clas, pan_range=(-1, 1), tilt_range=(-1, 1), zoom_range=(0, 1)):
    """
    Calculate new pan, tilt, and zoom values to center the object and ensure it fits within the frame vertically.

    :param object_center: Tuple (x, y) of the center of the object in the frame.
    :param object_bbox: Tuple (x1, y1, x2, y2) representing the bounding box of the object.
    :param frame_dimensions: Tuple (width, height) of the frame.
    :param current_pan: The current pan position of the camera.
    :param current_tilt: The current tilt position of the camera.
    :param pan_range: The allowed pan range for the camera (default: -1 to 1).
    :param tilt_range: The allowed tilt range for the camera (default: -1 to 1).
    :param zoom_range: The allowed zoom range for the camera (default: 0 to 1).

    :return: Tuple (new_pan, new_tilt, zoom_factor)
    """
    frame_height, frame_width  = frame_dimensions
    frame_center_x, frame_center_y = frame_width // 2, frame_height // 2
    object_center_x, object_center_y = object_center
    x1, y1, x2, y2 = object_bbox
    # print(object_bbox)
    print(f'{clas}, obj: {object_center_y}, frame {frame_center_y}')
    # Calculate the offset in pixels for pan and tilt
    x_offset = object_center_x - frame_center_x
    y_offset = frame_center_y - object_center_y
    tilt_per_frame = 0.49
    pan_per_frame = 0.42
    tilt_per_pixel = tilt_per_frame/frame_height
    pan_per_pixel = pan_per_frame/frame_width
    # Convert pixel offsets to pan/tilt adjustments
    # pan_adjustment = (x_offset/4742)
    # if clas == 2 or clas == 3:
    #     tilt_adjustment = (y_offset / 2948)
    # elif clas == 0 or clas == 1:
    #     tilt_adjustment = (y_offset / 2948)  # Adjust this value as needed
    pan_adjustment = (x_offset*pan_per_pixel)
    tilt_adjustment = (y_offset * tilt_per_pixel)  # Adjust this value as needed
    # Calculate the new pan and tilt values
    new_pan = current_pan + pan_adjustment
    new_tilt = current_tilt + tilt_adjustment

    # Zoom adjustment based on the vertical bounding box size of the object
    object_height = y2 - y1  # The height of the object in pixels
    # print(F"Potential problem {object_height} {frame_height}")
    # The desired vertical size in the frame (entire height of frame)
    zoom_ratio = object_height/frame_height
    # print("this is it" ,zoom_ratio)
    # Calculate zoom factor: If object is too big, zoom out; if too small, zoom in
    zoom_factor = zoom_ratio

    return new_pan, new_tilt, zoom_factor


def save_captured_image(frame, bounding_box, timestamp, object_type, output_dir="captured_images"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    x1, y1, x2, y2 = bounding_box
    height, width, _ = frame.shape
    # cropped_image = frame[int(y1) - 30:int(y2) + 3, int(x1) - 50:50 + int(x2)]
    x1 = max(0, x1 - 50)
    y1 = max(0, y1 - 30)
    x2 = min(width, x2 + 50)
    y2 = min(height, y2 + 3)

    cropped_image = frame[y1:y2, x1:x2]
    filename = f"{output_dir}/{object_type}_{timestamp}.png"

    cv2.imwrite(filename, cropped_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print(f"Saved {object_type} image as {filename}")


##################################################################################################################
####################################################preprocessing capture image for ocr#################################

def split_ventilator_image(image_path):
    """Splits a ventilator image into upper and lower parts and saves them with timestamps."""
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Define the upper and lower parts
    # upper_part = image[0:int(height * 0.25), 0:width]
    # lower_part = image[int(height * 0.5):int(height * 0.67), 0:width]
    upper_part = image[0:int(height * 0.3), 0:width]
    lower_part = image[int(height * 0.50):int(height), 0:width]
    # Save the parts with timestamps
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    img_parts_dir = r"C:\Users\OMEN\PycharmProjects\object_detection\yolov7\Vent_img_parts"
    if not os.path.exists(img_parts_dir):
        os.makedirs(img_parts_dir)
    upper_part_path = os.path.join(img_parts_dir, f"upper_part_{timestamp}.png")
    lower_part_path = os.path.join(img_parts_dir, f"lower_part_{timestamp}.png")

    cv2.imwrite(upper_part_path, upper_part, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    cv2.imwrite(lower_part_path, lower_part, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    logging.info(f"Upper part saved to {upper_part_path}")
    logging.info(f"Lower part saved to {lower_part_path}")
    destination_path = r"C:\Users\OMEN\PycharmProjects\object_detection\yolov7\logged_ventilator_images"
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
    shutil.move(image_path, destination_path)
    # Extract data from the saved parts
    upper_values = extract_data_from_image(upper_part, 'upper_part')
    lower_values = extract_data_from_image(lower_part, 'lower_part')

    return upper_values, lower_values


def process_paramonitor_image(image_path):
    """Processes the paramonitor image to extract values like Pulse rate, blood pressure, SpO2, and temperature."""
    image = cv2.imread(image_path)
    logging.info(f"Processing paramonitor image of size: {image.shape}")
    paramonitor_values = extract_data_from_image(image, 'paramonitor')
    destination_path = r"C:\Users\OMEN\PycharmProjects\object_detection\yolov7\logged_paramonitor_images"
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
    shutil.move(image_path, destination_path)
    # Log the extracted values or save them to Excel as needed
    return paramonitor_values


def process_yolo_images():
    image_directory = r'C:\Users\OMEN\PycharmProjects\object_detection\yolov7\captured_images'
    """Processes images saved by YOLO, distinguishing between ventilator and paramonitor types."""
    data = {}

    for image_filename in os.listdir(image_directory):
        if image_filename.endswith('.png'):
            image_path = os.path.join(image_directory, image_filename)
            label_ext = image_filename.split('.')[0].split('_')
            application_name, dates, times = label_ext
            # Format date as YYYY-MM-DD
            date_final = f"{dates[:4]}-{dates[4:6]}-{dates[6:]}"
            time_final = f"{times[:2]}:{times[2:4]}:{times[4:]}"

            # Check image type based on filename and call appropriate function
            if 'Ventilator' in image_filename:
                upper_values, lower_values = split_ventilator_image(image_path)
                logging.info(f"Extracted values from vent: {upper_values}, {lower_values}")
                data['Ventilator'] = {**upper_values, **lower_values}  # Dictionary merge for Python < 3.9
            elif 'Paramonitor' in image_filename:
                paramonitor_values = process_paramonitor_image(image_path)
                logging.info(f"Extracted values from paramonitor: {paramonitor_values}")
                data['Paramonitor'] = paramonitor_values
            else:
                logging.warning(f"Unknown image type for {image_filename}")
    return data, date_final, time_final, application_name


#######################################################

# Load the Qwen2VL model and processor
model_q = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
print("************************************ :", next(model_q.parameters()).is_cuda)


def extract_data_from_image(img_part, part_name):
    """Extracts specific parameter values from the ventilator image's parts using model_q."""

    # Define prompts based on part (upper or lower)
    if part_name == 'upper_part':
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text",
                     "text": "From the upper part of the ventilator image, what are the numerical values of PEAK, PMEAN, PEEP1, I:E, FTOT, VTE, VETOT ?"}
                ]
            }
        ]
    elif part_name == 'lower_part':
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text",
                     "text": "From the lower part of the ventilator image, what are the numerical values of PEEP2, VT, VMAX, O2?"}
                ]
            }
        ]
    elif part_name == 'paramonitor':
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text",
                     "text": "From the paramonitor image, what are the numerical values of Pulse Rate (PR), SpO2, Temperature (TEMP), and Blood Pressure ?"}
                ]
            }
        ]

    else:
        print("Data NOT Found")

    # Create the prompt and process the image using  Qwen model
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], images=[Image.fromarray(img_part)], padding=True, return_tensors="pt")
    inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")

    # Generate output
    output_ids = model_q.generate(**inputs, max_new_tokens=1024)
    output_text = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    logging.info(f"Recognized text for {part_name}: {output_text}")

    # Extract numerical values from the recognized text
    extracted_values = extract_values(output_text, part=part_name)
    return extracted_values


def extract_values(text, part):
    """
    Extracts specific parameter values from recognized text based on the image part.
    """

    extracted_values = {}

    # Define regex patterns for each part of the image
    if part == 'upper_part':
        patterns = {
            'PEAK': r'PEAK\s*:\s*(\d+\.?\d*)',
            'PMEAN': r'PMEAN\s*:\s*(\d+\.?\d*)',
            'PEEP1': r'PEEP1\s*:\s*(\d+\.?\d*)',
            'I:E': r'\bI:E\s*:\s*(\d+:\d+\.?\d*)',
            'FTOT': r'\bFTOT\s*:\s*(\d+\.?\d*)',
            'VTE': r'VTE\s*:\s*(\d+\.?\d*)',
            'VETOT': r'\bVETOT\s*:\s*(\d+\.?\d*)'

        }
    elif part == 'lower_part':
        patterns = {
            'PEEP2': r'\bPEEP2?\s*:\s*(\d+\.?\d*).*',
            'VT': r'VT\s*:\s*(\d+\.?\d*)',
            'O2': r'O2\s*:\s*(\d+\.?\d*)'
        }
    elif part == 'paramonitor':

        patterns = {
            r"\**Pulse Rate \((PR)\)\**: (\d+)",
            r"\**(SpO2)\**\s*[:\-]?\s*(\d+)%?",
            r"\**Temperature \((TEMP)\)\**: (\d+\.\d+)",
            r"\**Blood Pressure \((BP|NIBP)\)\**: (\d+\/\d+)"
        }
    else:
        logging.warning(f"Unknown part: {part}")
        return extracted_values  # Return empty dictionary if part is unknown

    if part == 'paramonitor':
        # Extract values using regex
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                extracted_values[match.group(1)] = match.group(2)  # Group(1) contains the numeric value
                print(f"Extracted {match.group(1)}: {extracted_values[match.group(1)]}")
            else:
                try:
                    print(f"No match found for {match.group(1)}")
                except AttributeError:
                    print('New Pattern needed!!')
                    continue
    else:
        for keys, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                extracted_values[keys] = match.group(1)  # Group(1) contains the numeric value
                print(f"Extracted {keys}: {extracted_values[keys]}")
            else:
                print(f"No match found for {keys}")

    return extracted_values


#####################################################
# Now Save the Ventilator and Paramonitor data seperately in 2 different excel files


def save_data_to_excel(data, current_date, current_time, application):
    if data:
        try:
            ventilator_data = data['Ventilator']
        except:
            paramonitor_data = data['Paramonitor']

        save_time = {"Date": current_date, "Time": current_time}
        print(save_time)
        if not os.path.exists(excel_output_dir):
            os.makedirs(excel_output_dir)
        try:
            rows = {**save_time, **ventilator_data}
            print(rows)
            ventilator_columns = ['Date', 'Time', 'PEAK', 'PMEAN', 'PEEP1', 'I:E', 'FTOT', 'VTE', 'VETOT', 'PEEP2',
                                  'VT', 'O2']
            df = pd.DataFrame([rows], columns=ventilator_columns)
            df = df[['Date', 'Time', 'PEAK', 'PMEAN', 'PEEP1', 'I:E', 'FTOT', 'VTE', 'VETOT', 'PEEP2', 'VT', 'O2']]
            data_path = os.path.join(excel_output_dir, 'ventilator.csv')
            logging.info(f"Excel File updated at: {data_path}")
            if os.path.exists(data_path):
                repeat = pd.read_csv(data_path)
                print(
                    'Is is comeing here ######################################################################################')
                print('Checkin: ', repeat['Time'].iloc[-1])
                print(df['Time'].iloc[-1])
                print(
                    'Is is comeing here ######################################################################################')
                if repeat['Time'].iloc[-1] == df['Time'].iloc[-1]:
                    pass
                else:
                    df.to_csv(data_path, mode='a', header=False, index=False)
            else:
                df.to_csv(data_path, header=True, index=False)
            log_action(f'{application} data extracted and excel file updated.')
            return {'Ventilator': df}
        except:
            rows = {**save_time, **paramonitor_data}
            paramonitor_columns = ['Date', 'Time', 'PR', 'SpO2', 'TEMP', 'BP']
            if 'NIBP' in [*rows]:
                rows.update({'BP': rows.pop('NIBP')})

            # df = pd.DataFrame([rows], columns=paramonitor_columns)
            # df1 = df[['Date', 'Time','PR', 'SpO2', 'TEMP', 'BP']]
            # data_path = os.path.join( excel_output_dir, 'paramonitor.csv')
            # logging.info(f"Excel File updated at: {data_path} ")
            # if os.path.exists(data_path):
            #     df1.to_csv(data_path, mode='a', header=False)
            # else:
            #     df1.to_csv(data_path, header=True)
            # return {'Paramonitor' : df}
            rows = {key: rows.get(key, None) for key in paramonitor_columns}

            df = pd.DataFrame([rows], columns=paramonitor_columns)
            data_path = os.path.join(excel_output_dir, 'paramonitor.csv')
            logging.info(f"Excel File updated at: {data_path}")
            if os.path.exists(data_path):
                df.to_csv(data_path, mode='a', header=False, index=False)
            else:
                df.to_csv(data_path, header=True, index=False)
            log_action(f'{application} data extracted and excel file updated.')
            return {'Paramonitor': df}


def seconds_until(time_str):
    # Parse the input time (e.g., "16:03")
    target_time = datetime.strptime(time_str, "%H:%M").time()

    now = datetime.now()
    today_target = datetime.combine(now.date(), target_time)

    # If the target time has already passed today, assume it's for tomorrow
    if today_target <= now:
        today_target += timedelta(days=1)

    seconds_left = int((today_target - now).total_seconds())
    return seconds_left


####################################################################################################################
# logs module
today = datetime.now()
file_start_date = today - timedelta(days=1)


def process_date(date):
    return date.strftime("%d/%m/%Y"), date.strftime("%H:%M:%S")


logs = {
    'Date': None,
    'Start_time': None,
    'Last_capture': None,
    'Device_usage_count': 0,
    'Data_capture_count': 0,
    'total_duration': "0hr 0min 0sec",
    'longest_duration': "0hr 0min 0sec",
    'No_of_person': 0
}
device_switch = False
final_count = 0
todays_count = 0
longest_duration = 0
total_duration = 0
duration_start = 0
on_here = False
previous_count = 0
previous_frame = False
tracking_on = False


def restart_application():
    """Restart the entire application."""
    print("Restarting the system...")
    python = sys.executable
    os.execl(python, python, *sys.argv)


def convert_seconds(seconds):
    hours, remainder = divmod(seconds, 3600)  # Get hours & remaining seconds
    minutes, seconds = divmod(remainder, 60)  # Get minutes & remaining seconds
    return hours, minutes, seconds


def check_midnight():
    global logs
    global final_count
    global todays_count
    global longest_duration
    global total_duration
    global duration_start
    """Background task to check for midnight and save frames."""
    while True:
        now = datetime.now().strftime("%H:%M:%S")
        logs['Device_usage_count'] = final_count
        logs['Data_capture_count'] = todays_count
        logs['longest_duration'] = longest_duration
        logs['total_duration'] = total_duration
        if now == '00:00:00':
            hr, mins, sec = convert_seconds(int(logs['longest_duration']))
            logs['longest_duration'] = str(hr) + 'hr ' + str(mins) + 'min ' + str(sec) + 'sec'
            hr, mins, sec = convert_seconds(int(logs['total_duration']))
            logs['total_duration'] = str(hr) + 'hr ' + str(mins) + 'min ' + str(sec) + 'sec'
            print("It's midnight! Saving data.")
            data = pd.DataFrame([logs])
            data_path = r'C:\Users\OMEN\PycharmProjects\object_detection\yolov7\excel_output\logs.csv'
            logging.info(f"Excel File updated at: {data_path}")
            if os.path.exists(data_path):
                if not all(value in [None, "", 0, "0hr 0min 0sec"] for value in logs.values()):
                    data = pd.DataFrame([logs])
                    data.to_csv(data_path, mode='a', header=False, index=False)
            else:
                if not all(value in [None, "", 0, "0hr 0min 0sec"] for value in logs.values()):
                    data.to_csv(data_path, header=True, index=False)
            logs = {
                'Date': None,
                'Start_time': None,
                'Last_capture': None,
                'Device_usage_count': 0,
                'Data_capture_count': 0,
                'total_duration': "0hr 0min 0sec",
                'longest_duration': "0hr 0min 0sec",
                'No_of_person': 0
            }
            final_count = 0
            todays_count = 0
            longest_duration = 0
            total_duration = 0
            duration_start = 0
            time.sleep(1)  # Prevent multiple saves in the same second
        time.sleep(0.1)


LOG_FILE = r'C:\Users\OMEN\PycharmProjects\object_detection\yolov7\excel_output\daily_log.txt'
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)  # Ensure log directory exists

# ✅ Terminal Logger (default Flask logging)
logging.basicConfig(
    level=logging.INFO,  # ✅ Logs to terminal
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ✅ Separate File Logger (Only for log_action)
file_logger = logging.getLogger("FileLogger")
file_logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
file_logger.addHandler(file_handler)


def log_action(action_text):
    """Log an action to a file but not the terminal."""
    file_logger.info(action_text)  # ✅ Saves onl


def tail(file_path, lines=10):
    """ Read the last `lines` lines of a text file """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            last_lines = f.readlines()[-lines:]  # Read last 10 lines
            return list(reversed(last_lines))  # Reverse them before sending
    except FileNotFoundError:
        return ["No logs available."]


@app.route('/get_log_data')
def get_log_data():
    data_path = os.path.join(excel_output_dir, 'daily_log.txt')
    if os.path.exists(data_path):
        last_lines = tail(data_path)
        return jsonify({"logs": [line.strip() for line in last_lines]})  # ✅ Remove trailing newlines
    return jsonify({"logs": []})


####################################################################################################################
# # model_path1 = r'C:\Users\OMEN\PycharmProjects\object_detection\yolov7\runs\combined\exp\weights\best.pt'
# model_path1 = r"C:\Users\OMEN\PycharmProjects\object_detection\yolov7\runs\aiims_weights\aiim_yolo7.pt"
# model_event_tracking = torch.hub.load("WongKinYiu/yolov7", "custom", model_path1, trust_repo=True)
# model_event_tracking = model_event_tracking.to('cuda')
# print("************************************ :", next(model_event_tracking.parameters()).is_cuda)
# Example class mapping (adjust this according to your model's classes)
# CLASS_NAMES = {0: "BP_measured", 1: "Hand_Sanitized", 2: "Apron", 3: "no_head_cover", 4: 'Oxygen_connected',
#                5: 'Face', 6: 'Paramonitor', 7: 'Paramonitor_off', 8:'Human', 9: 'Ventilator', 10: 'Ventilator_off'}


CLASS_NAMES = {0: "Apron", 1: "Bed_No", 2: "Board", 3: "Human", 4: "Face", 5: "head_cover",6:"mask",7: "no_head_cover",8: "no_mask",9:"no_shoe_cover",10: "Paramonitor",11: "Patient_face",12:"shoe_cover", 13:"Ventilator"}
screens = ['Paramonitor','Paramonitor_off', 'Ventilator','Ventilator_off']
off_screens = ['Paramonitor_off', 'Ventilator_off']

rotation1 = 0
detected1 = []
capture1 = []
ground1 = 0
model = YOLO(r"C:\Users\OMEN\PycharmProjects\object_detection\yolov7\runs\aiims_weights\aiims_yolov8.pt")

def generate_frames():
    global logs
    camera_ip = "10.2.37.16"
    camera_port = 2000
    username = "admin"
    password = "admin"
    global tracking_on
    tracking_on = False
    # camera = cv2.VideoCapture('rtsp://10.2.37.16:554/live/av0')
    # Initialize camera connection
    camera1 = ONVIFCamera(camera_ip, camera_port, username, password)
    ptz_service = camera1.create_ptz_service()
    media_service = camera1.create_media_service()
    profiles = media_service.GetProfiles()
    profile = profiles[0]
    # pan, tilt = camera_state['pan'], camera_state['tilt']
    pan = 0
    tilt = -0.276
    zoom = 0
    previous_round = 0
    move_and_zoom_camera(ptz_service, profile, pan, tilt, 0.0)
    t1 = time.time()
    t2 = time.time()
    rotation = 0
    detected = []
    capture = []
    ground = 0
    global rotation1
    global detected1
    global capture1
    global ground1
    global action
    global bed_no
    global bed_to_pan
    while True:
        t1 = time.time()
        t2 = time.time()
        if action == 'start':
            monitor_time = seconds_until(monitor_till)
            pan = bed_to_pan[bed_no]
            action = 'hold'
        elif action == 'cancel':
            monitor_time = 14
        cover_timer = time.time()
        mask_timer = time.time()
        mask_1 = True
        cover = True
        while t2 - t1 < 10:
            # print("Loop 1")
            move_and_zoom_camera(ptz_service, profile, pan, tilt, 0.0)
            success, new_frame = camera.read()
            if not success:
                print("Failed to grab frame")
                break

            #img = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
            results = model(new_frame, verbose=False, conf=0.1)
            bounding_boxes = results[0].boxes.data.to('cpu').numpy()

            new_bounding_boxes = []

            for box in bounding_boxes:                                      #Keeping only the top detection for person and labcoat
                x1, y1, x2, y2, confidence, class_id = box[:6]
                class_id = int(class_id)
                if class_id in [0, 3] and confidence < .30:
                    continue
                new_bounding_boxes.append([x1, y1, x2, y2, confidence, class_id])
            class_ids = [box[-1] for box in new_bounding_boxes]
            labels0 = [CLASS_NAMES[i] for i in class_ids]                      #listing all the predicted object
            # if 'Human' not in labels0 and 'Human' in detected:
            #     if "Hand_Sanitized" not in detected:
            #         socketio.emit('notification', {'message': f'Hand was not sanitized at {detected[0]}!'})
            #         log_action('Hand was not sanitized.')
            #     if "BP_measured" not in detected:
            #         socketio.emit('notification', {'message': f'BP was not Measured at {detected[0]}!'})
            #         log_action('BP was not measured.')
            timer2 = time.time()
            if ('no_head_cover' in labels0 and timer2 - cover_timer >8) or ('no_head_cover' in labels0 and cover):
                socketio.emit('notification', {'message': f'Visitor without head cover at {datetime.now().time().strftime("%H:%M:%S")}!'})
                log_action('Visitor was not wearing headcover.')
                cover_timer = time.time()
                cover = False
            if ('no_mask' in labels0 and timer2 - mask_timer >8) or ('no_mask' in labels0 and mask_1) :
                socketio.emit('notification', {'message': f'Visitor without mask at {datetime.now().time().strftime("%H:%M:%S")}!'})
                log_action('Visitor was not wearing mask.')
                mask_timer = time.time()
                mask_1 = False
            for box in new_bounding_boxes:                                      #looping through the
                x1, y1, x2, y2 = box[:4] # Convert coordinates to integers
                confidence = box[4]
                class_id = int(box[5])
                label = CLASS_NAMES.get(class_id, "Unknown")
                # print(label, confidence)
                if label not in screens:
                    if label != 'Face' and label != 'Patient_face':
                        if 'Human' not in labels0:
                            ground1 = 0
                        if 'Human' in labels0 and "Apron" not in labels0:
                            now = datetime.now().time()
                            time_stamp = now.strftime("%H:%M:%S")
                            detected1.insert(0, time_stamp)
                            detected1.append('Human')

                            if ground1 == 0:
                                socketio.emit('notification', {'message': 'Person without Apron detected!'})
                                log_action('Person without Apron detected')
                                ground1 = 1
                            if label == 'Human':
                                cv2.rectangle(new_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                                cv2.putText(new_frame, f"{label} ({confidence:.2f})", (int(x1), int(y1) - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            elif label == 'no_head_cover':
                                detected1.append(label)
                                cv2.rectangle(new_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                                cv2.putText(new_frame, f"{label} ({confidence:.2f})", (int(x1), int(y1) - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, (0, 0, 255), 2)
                            else:
                                detected1.append(label)
                                cv2.rectangle(new_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                                cv2.putText(new_frame, f"{label} ({confidence:.2f})", (int(x1), int(y1) - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, (0, 255, 0), 2)
                        else:
                            if label == 'Human':
                                now = datetime.now().time()
                                time_stamp = now.strftime("%H:%M:%S")
                                detected1.insert(0, time_stamp)
                                detected1.append('Human')
                            if label == 'no_head_cover':
                                detected1.append(label)
                                cv2.rectangle(new_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                                cv2.putText(new_frame, f"{label} ({confidence:.2f})", (int(x1), int(y1) - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, (0, 0, 255), 2)
                            else:
                                detected1.append(label)
                                cv2.rectangle(new_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                                cv2.putText(new_frame, f"{label} ({confidence:.2f})", (int(x1), int(y1) - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, (0, 255, 0), 2)
                    elif label == 'Face' and privacy_mode:
                        try:
                            roi = new_frame[int(y1) - 20:int(y2) + 3, int(x1) - 20:20 + int(x2)]
                            roi = cv2.GaussianBlur(roi, (99, 99), 70)
                            new_frame[int(y1) - 20:int(y2) + 3, int(x1) - 20:20 + int(x2)] = roi
                        except cv2.error:
                            roi = new_frame[int(y1):int(y2), int(x1):int(x2)]
                            roi = cv2.GaussianBlur(roi, (99, 99), 70)
                            new_frame[int(y1):int(y2), int(x1):int(x2)] = roi
                    elif label == 'Patient_face' and confidence >.4 and patient_privacy_mode:
                        try:
                            roi = new_frame[int(y1) - 20:int(y2) + 3, int(x1) - 20:20 + int(x2)]
                            # print("here1")
                            roi = cv2.GaussianBlur(roi, (99, 99), 70)
                            new_frame[int(y1) - 20:int(y2) + 3, int(x1) - 20:20 + int(x2)] = roi
                        except cv2.error:
                            roi = new_frame[int(y1):int(y2), int(x1):int(x1)]
                            # print("Here2")
                            roi = cv2.GaussianBlur(roi, (99, 99), 70)
                            new_frame[int(y1):int(y2), int(x1):int(x2)] = roi
                        finally:
                            continue
                elif label in screens and confidence >.5:
                    if class_id in [13,9]:
                        cv2.rectangle(new_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(new_frame, f"{label} ({confidence:.2f})", (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        cv2.rectangle(new_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        cv2.putText(new_frame, f"{label} ({confidence:.2f})", (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', new_frame)
            new_frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + new_frame + b'\r\n')
            t2 = time.time()

        start_time1 = time.time()
        start_time2 = time.time()
        while start_time2-start_time1 < monitor_time:
            if action == 'start':
                break
            elif action =='cancel' and previous_round != 0:
                previous_round = 0
                break
            # print("Loop 2")
            previous_round += 1
            rotation1 = 0
            detected1 = []
            capture1 = []
            ground1 = 0
            # print('Loop 2 has started!!')
            success, frame = camera.read()
            if not success:
                print("Failed to grab frame")
                break
            # print(frame.size)

            # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame, conf=0.1)
            bounding_boxes = results[0].boxes.data.to('cpu').numpy()
            new_bounding_boxes = []
            for box1 in bounding_boxes:
                x1, y1, x2, y2, confidence, class_id = box1[:6]
                class_id1 = int(class_id)
                if class_id1 in [0, 3] and confidence < .30:
                    continue
                new_bounding_boxes.append([x1, y1, x2, y2, confidence, class_id])
            # print(bounding_boxes)
            class_ids = [box[-1] for box in new_bounding_boxes]
            labels1 = [CLASS_NAMES[i] for i in class_ids]
            if 'Human' not in labels1 and 'Human' in detected:
                # if "Hand_Sanitized" not in detected:
                #     socketio.emit('notification', {'message': f'Hand was not sanitized at {detected[0]}!'})
                #     log_action('Hand was not sanitized.')
                # if "BP_measured" not in detected:
                #     socketio.emit('notification', {'message': f'BP was not Measured at {detected[0]}!'})
                #     log_action('BP was not measured.')
                detected = []
            timer2 = time.time()
            if 'no_head_cover' in labels1 and timer2 - cover_timer > 8:
                socketio.emit('notification', {'message': f'Visitor without head cover at {datetime.now().time().strftime("%H:%M:%S")}!'})
                log_action('Visitor was not wearing headcover.')
                cover_timer = time.time()
            if 'no_mask' in labels1 and timer2 - mask_timer > 8:
                socketio.emit('notification', {'message': f'Visitor without mask at {datetime.now().time().strftime("%H:%M:%S")}!'})
                log_action('Visitor was not wearing mask.')
                mask_timer = time.time()

            for box1 in new_bounding_boxes:
                x1, y1, x2, y2 = box1[:4]  # Convert coordinates to integers
                confidence_main = box1[4]
                class_id1 = int(box1[5])
                label_main = CLASS_NAMES.get(class_id1, "Unknown")
                if 'Human' not in labels1:
                    ground = 0
                if 'Human' in labels1 and "Apron" not in labels1:
                    now = datetime.now().time()
                    time_stamp = now.strftime("%H:%M:%S")
                    detected.insert(0, time_stamp)
                    detected.append('Human')

                    if ground == 0:
                        socketio.emit('notification', {'message': 'Person without apron detected!'})
                        log_action('Person without apron detected')
                        ground = 1

                if label_main in screens and confidence_main > .5:
                    object_center = [(x2 + x1) // 2, (y2 + y1) // 2]
                    # print(object_center)
                    frame_dimension = frame.shape[:2]
                    # print(frame_dimension)
                    # status = ptz_service.GetStatus({'ProfileToken': profile.token})
                    # original_pan = status.Position.PanTilt.x
                    # original_tilt = status.Position.PanTilt.y
                    original_pan = pan
                    original_tilt = tilt
                    print("Original after change", original_pan)
                    print(box1[:4])
                    new_pan, new_tilt, new_zoom = calculate_new_pan_tilt_and_zoom(object_center, box1[:4],
                                                                                  frame_dimension, original_pan,
                                                                                  original_tilt, class_id1)
                    # print(new_pan, new_tilt, new_zoom)
                    # status = ptz_service.GetStatus({'ProfileToken': profile.token})
                    # current_pan = status.Position.PanTilt.x
                    # current_tilt = status.Position.PanTilt.y
                    if new_pan < -1:
                        new_pan = -1
                    elif new_pan > 1:
                        new_pan = 1

                    move_and_zoom_camera(ptz_service, profile, new_pan, new_tilt, 0)

                    t1 = time.time()
                    t2 = time.time()
                    while t2 - t1 <2:
                        if action == 'start':
                            break
                        success, new_frame = camera.read()
                        if not success:
                            print("Failed to grab frame")
                            break
                        # print(frame.size)

                        #img = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
                        results = model(new_frame, conf=0.1)
                        new_bounding_boxes1 = results[0].boxes.data.to('cpu').numpy()
                        new_bounding_boxes2 = []
                        for box in new_bounding_boxes1:
                            x3, y3, x4, y4, confidence, class_id = box[:6]
                            class_id = int(class_id)
                            if class_id in [0, 3] and confidence < .30:
                                continue
                            new_bounding_boxes2.append([x3, y3, x4, y4, confidence, class_id])
                        # print(new_bounding_boxes)
                        class_ids = [box[-1] for box in new_bounding_boxes2]
                        labels2 = [CLASS_NAMES[i] for i in class_ids]
                        timer2 = time.time()
                        if 'no_head_cover' in labels2 and timer2 - cover_timer > 8:
                            socketio.emit('notification',
                                          {'message': f'Visitor without head cover at {datetime.now().time().strftime("%H:%M:%S")}!'})
                            log_action('Visitor was not wearing headcover.')
                            cover_timer = time.time()
                        if 'no_mask' in labels2 and timer2 - mask_timer > 8:
                            socketio.emit('notification',
                                          {'message': f'Visitor without mask at {datetime.now().time().strftime("%H:%M:%S")}!'})
                            log_action('Visitor was not wearing mask.')
                            mask_timer = time.time()
                        for box in new_bounding_boxes2:
                            x3, y3, x4, y4 = box[:4]  # Convert coordinates to integers
                            confidence = box[4]
                            class_id = int(box[5])
                            label = CLASS_NAMES.get(class_id, "Unknown")
                            if class_id == class_id1:
                                object_center = [(x3 + x4) // 2, (y3 + y4) // 2]
                                frame_dimension = new_frame.shape[:2]
                                _, new_tilt1, new_zoom = calculate_new_pan_tilt_and_zoom(object_center, box[:4],
                                                                                              frame_dimension,
                                                                                              new_pan,
                                                                                              new_tilt, class_id1)
                                print(new_tilt, new_tilt1, "thisisisisisiis sisisiis")
                            if label not in screens:
                                if label == 'Face' and privacy_mode:
                                    try:
                                        roi = new_frame[int(y3) - 20:int(y4) + 3, int(x3) - 20:20 + int(x4)]
                                        roi = cv2.GaussianBlur(roi, (99, 99), 70)
                                        new_frame[int(y3) - 20:int(y4) + 3, int(x3) - 20:20 + int(x4)] = roi
                                    except cv2.error:
                                        roi = new_frame[int(y3):int(y4), int(x3):int(x4)]
                                        roi = cv2.GaussianBlur(roi, (99, 99), 70)
                                        new_frame[int(y3):int(y4), int(x3):int(x4)] = roi
                                elif label == 'Patient_face' and confidence >.4 and patient_privacy_mode:
                                    try:
                                        roi = new_frame[int(y3) - 20:int(y4) + 3, int(x3) - 20:20 + int(x4)]
                                        roi = cv2.GaussianBlur(roi, (99, 99), 70)
                                        new_frame[int(y3) - 20:int(y4) + 3, int(x3) - 20:20 + int(x4)] = roi
                                    except cv2.error:
                                        roi = new_frame[int(y3):int(y4), int(x3):int(x4)]
                                        roi = cv2.GaussianBlur(roi, (99, 99), 70)
                                        new_frame[int(y3):int(y4), int(x3):int(x4)] = roi
                                else:
                                    if 'Apron' in labels2:
                                        if label == "no_head_cover":
                                            cv2.rectangle(new_frame, (int(x3), int(y3)), (int(x4), int(y4)),
                                                          (0, 0, 255), 2)
                                            cv2.putText(new_frame, f"{label} ({confidence:.2f})",
                                                        (int(x3), int(y3) - 10),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                        else:
                                            cv2.rectangle(new_frame, (int(x3), int(y3)), (int(x4), int(y4)),
                                                          (0, 255, 0), 2)
                                            cv2.putText(new_frame, f"{label} ({confidence:.2f})",
                                                        (int(x3), int(y3) - 10),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                    else:
                                        if label in ["BP_measured", "Hand_Sanitized"]:
                                            cv2.rectangle(new_frame, (int(x3), int(y3)), (int(x4), int(y4)),
                                                          (0, 255, 0), 2)
                                            cv2.putText(new_frame, f"{label} ({confidence:.2f})",
                                                        (int(x3), int(y3) - 10),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                        else:
                                            cv2.rectangle(new_frame, (int(x3), int(y3)), (int(x4), int(y4)),
                                                          (0, 0, 255), 2)
                                            cv2.putText(new_frame, f"{label} ({confidence:.2f})",
                                                        (int(x3), int(y3) - 10),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            elif label == label_main and confidence >.8:
                                print(f"Label and main {label}, {label_main}")
                                frame_dimension = new_frame.shape[:2]
                                frame_height, frame_width = frame_dimension
                                object_height = y4 - y3  # The height of the object in pixels
                                # print(F"Potential problem {object_height} {frame_height}")
                                # The desired vertical size in the frame (entire height of frame)
                                new_zoom1 = object_height / frame_height
                                if label in ['Paramonitor', 'Ventilator']:
                                    cv2.rectangle(new_frame, (int(x3), int(y3)), (int(x4), int(y4)), (0, 255, 0), 2)
                                    cv2.putText(new_frame, f"{label} ({confidence:.2f})", (int(x3), int(y3) - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                else:
                                    cv2.rectangle(new_frame, (int(x3), int(y3)), (int(x4), int(y4)), (0, 0, 255), 2)
                                    cv2.putText(new_frame, f"{label} ({confidence:.2f})", (int(x3), int(y3) - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        ret, buffer = cv2.imencode('.jpg', new_frame)
                        new_frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + new_frame + b'\r\n')
                        t2 = time.time()


                    print(1 - new_zoom)
                    print(1 - (new_zoom - 0.06))
                    print(class_id1)
                    if class_id1 == 9:
                        move_and_zoom_camera(ptz_service, profile, new_pan, new_tilt1 + 0.033,
                                             (1 - (new_zoom - 0.12)))
                        # if original_pan == 0:
                        #     move_and_zoom_camera(ptz_service, profile, new_pan, new_tilt+0.04,
                        #                          (1 - (new_zoom - 0.06)))
                        # else:
                        #     move_and_zoom_camera(ptz_service, profile, new_pan, new_tilt + 0.065,
                        #                          (1 - (new_zoom - 0.09)))
                    else:
                        move_and_zoom_camera(ptz_service, profile, new_pan, new_tilt1 + 0.02, (1 - new_zoom))

                    while t2 - t1 < 8:
                        if action == 'start':
                            break
                        success, new_frame = camera.read()
                        if not success:
                            print("Failed to grab frame")
                            break
                        # print(frame.size)

                        #img = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
                        results = model(new_frame, conf=0.1)
                        new_bounding_boxes1 = results[0].boxes.data.to('cpu').numpy()
                        new_bounding_boxes2 = []
                        for box in new_bounding_boxes1:
                            x3, y3, x4, y4, confidence, class_id = box[:6]
                            class_id = int(class_id)
                            if class_id in [0, 3] and confidence < .30:
                                continue
                            new_bounding_boxes2.append([x3, y3, x4, y4, confidence, class_id])
                        # print(new_bounding_boxes)
                        class_ids = [box[-1] for box in new_bounding_boxes2]
                        labels2 = [CLASS_NAMES[i] for i in class_ids]
                        timer2 = time.time()
                        if 'no_head_cover' in labels2 and timer2 - cover_timer > 8:
                            socketio.emit('notification',
                                          {'message': f'Visitor without head cover at {datetime.now().time().strftime("%H:%M:%S")}!'})
                            log_action('Visitor was not wearing headcover.')
                            cover_timer = time.time()
                        if 'no_mask' in labels2 and timer2 - mask_timer > 8:
                            socketio.emit('notification',
                                          {'message': f'Visitor without mask at {datetime.now().time().strftime("%H:%M:%S")}!'})
                            log_action('Visitor was not wearing mask.')
                            mask_timer = time.time()
                        for box in new_bounding_boxes2:
                            x3, y3, x4, y4 = box[:4]  # Convert coordinates to integers
                            confidence = box[4]
                            class_id = int(box[5])
                            label = CLASS_NAMES.get(class_id, "Unknown")
                            if label not in screens:
                                if label == 'Face' and privacy_mode:
                                    try:
                                        roi = new_frame[int(y3) - 20:int(y4) + 3, int(x3) - 20:20 + int(x4)]
                                        roi = cv2.GaussianBlur(roi, (99, 99), 70)
                                        new_frame[int(y3) - 20:int(y4) + 3, int(x3) - 20:20 + int(x4)] = roi
                                    except cv2.error:
                                        roi = new_frame[int(y3):int(y4), int(x3):int(x4)]
                                        roi = cv2.GaussianBlur(roi, (99, 99), 70)
                                        new_frame[int(y3):int(y4), int(x3):int(x4)] = roi
                                elif label == 'Patient_face' and confidence >.4 and patient_privacy_mode:
                                    try:
                                        roi = new_frame[int(y3) - 20:int(y4) + 3, int(x3) - 20:20 + int(x4)]
                                        roi = cv2.GaussianBlur(roi, (99, 99), 70)
                                        new_frame[int(y3) - 20:int(y4) + 3, int(x3) - 20:20 + int(x4)] = roi
                                    except cv2.error:
                                        roi = new_frame[int(y3):int(y4), int(x3):int(x4)]
                                        roi = cv2.GaussianBlur(roi, (99, 99), 70)
                                        new_frame[int(y3):int(y4), int(x3):int(x4)] = roi
                                else:
                                    if 'Apron' in labels2:
                                        if label == "no_head_cover":
                                            cv2.rectangle(new_frame, (int(x3), int(y3)), (int(x4), int(y4)),
                                                          (0, 0, 255), 2)
                                            cv2.putText(new_frame, f"{label} ({confidence:.2f})",
                                                        (int(x3), int(y3) - 10),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                        else:
                                            cv2.rectangle(new_frame, (int(x3), int(y3)), (int(x4), int(y4)),
                                                          (0, 255, 0), 2)
                                            cv2.putText(new_frame, f"{label} ({confidence:.2f})",
                                                        (int(x3), int(y3) - 10),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                    else:
                                        if label in ["BP_measured", "Hand_Sanitized"]:
                                            cv2.rectangle(new_frame, (int(x3), int(y3)), (int(x4), int(y4)),
                                                          (0, 255, 0), 2)
                                            cv2.putText(new_frame, f"{label} ({confidence:.2f})",
                                                        (int(x3), int(y3) - 10),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                        else:
                                            cv2.rectangle(new_frame, (int(x3), int(y3)), (int(x4), int(y4)),
                                                          (0, 0, 255), 2)
                                            cv2.putText(new_frame, f"{label} ({confidence:.2f})",
                                                        (int(x3), int(y3) - 10),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            elif label in screens and confidence > .8:

                                if label in ['Paramonitor', 'Ventilator']:
                                    cv2.rectangle(new_frame, (int(x3), int(y3)), (int(x4), int(y4)), (0, 255, 0), 2)
                                    cv2.putText(new_frame, f"{label} ({confidence:.2f})", (int(x3), int(y3) - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                else:
                                    cv2.rectangle(new_frame, (int(x3), int(y3)), (int(x4), int(y4)), (0, 0, 255), 2)
                                    cv2.putText(new_frame, f"{label} ({confidence:.2f})", (int(x3), int(y3) - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        ret, buffer = cv2.imencode('.jpg', new_frame)
                        new_frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + new_frame + b'\r\n')
                        t2 = time.time()
                    success, new_frame = camera.read()
                    if not success:
                        print("Failed to grab frame")
                        break
                    # print(frame.size)

                    #img = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
                    results = model(new_frame, conf=0.1)
                    new_bounding_boxes1 = results[0].boxes.data.to('cpu').numpy()
                    new_bounding_boxes2 = []
                    for box in new_bounding_boxes1:
                        x3, y3, x4, y4, confidence, class_id = box[:6]
                        class_id = int(class_id)
                        if class_id in [0, 3] and confidence < .30:
                            continue
                        new_bounding_boxes2.append([x3, y3, x4, y4, confidence, class_id])
                    # print(new_bounding_boxes)
                    class_ids = [box[-1] for box in new_bounding_boxes2]
                    labels3 = [CLASS_NAMES[i] for i in class_ids]
                    timer2 = time.time()
                    if 'no_head_cover' in labels3 and timer2 - cover_timer > 8:
                        socketio.emit('notification',
                                      {'message': f'Visitor without head cover at {datetime.now().time().strftime("%H:%M:%S")}!'})
                        log_action('Visitor was not wearing headcover.')
                        cover_timer = time.time()
                    if 'no_mask' in labels3 and timer2 - mask_timer > 8:
                        socketio.emit('notification',
                                      {'message': f'Visitor without mask at {datetime.now().time().strftime("%H:%M:%S")}!'})
                        log_action('Visitor was not wearing mask.')
                        mask_timer = time.time()
                    for box in new_bounding_boxes2:
                        x3, y3, x4, y4 = box[:4]  # Convert coordinates to integers
                        confidence = box[4]
                        class_id = int(box[5])
                        label = CLASS_NAMES.get(class_id, "Unknown")
                        if label in screens:
                            if confidence > .8 and class_id in (6, 9):
                                timestamp = time.strftime("%Y%m%d_%H%M%S")
                                # current_date = datetime.now().strftime('%Y-%m-%d')
                                # current_time = datetime.now().strftime('%H:%M:%S')
                                if class_id == 13:
                                    object_type = 'Ventilator'
                                    # save_captured_image(new_frame, (int(x3), int(y3), int(x4), int(y4)), timestamp,object_type)
                                elif class_id == 10:
                                    object_type = 'Paramonitor'
                                print((x3, y3, x4, y4))
                                save_captured_image(new_frame, (int(x3), int(y3), int(x4), int(y4)), timestamp, object_type)
                                cv2.rectangle(new_frame, (int(x3), int(y3)), (int(x4), int(y4)), (0, 255, 0), 2)
                                cv2.putText(new_frame, f"{label} ({confidence:.2f})", (int(x3), int(y3) - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            elif class_id in (15, 16):
                                if class_id == 10:
                                    object_type = 'Ventilator'
                                elif class_id == 7:
                                    object_type = 'Paramonitor'
                                log_action(f'Closed {object_type} detected!')
                                cv2.rectangle(new_frame, (int(x3), int(y3)), (int(x4), int(y4)), (0, 0, 255), 2)
                                cv2.putText(new_frame, f"{label} ({confidence:.2f})", (int(x3), int(y3) - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        elif label not in screens:
                            if label == 'Face' and privacy_mode:
                                try:
                                    roi = new_frame[int(y3) - 20:int(y4) + 3, int(x3) - 20:20 + int(x4)]
                                    roi = cv2.GaussianBlur(roi, (99, 99), 70)
                                    new_frame[int(y3) - 20:int(y4) + 3, int(x3) - 20:20 + int(x4)] = roi
                                except cv2.error:
                                    roi = new_frame[int(y3):int(y4), int(x3):int(x4)]
                                    roi = cv2.GaussianBlur(roi, (99, 99), 70)
                                    new_frame[int(y3):int(y4), int(x3):int(x4)] = roi
                            elif label == 'Patient_face' and confidence >.4 and patient_privacy_mode:
                                try:
                                    roi = new_frame[int(y3) - 20:int(y4) + 3, int(x3) - 20:20 + int(x4)]
                                    roi = cv2.GaussianBlur(roi, (99, 99), 70)
                                    new_frame[int(y3) - 20:int(y4) + 3, int(x3) - 20:20 + int(x4)] = roi
                                except cv2.error:
                                    roi = new_frame[int(y3):int(y4), int(x3):int(x4)]
                                    roi = cv2.GaussianBlur(roi, (99, 99), 70)
                                    new_frame[int(y3):int(y4), int(x3):int(x4)] = roi
                            else:
                                if 'Apron' in labels3:
                                    if label == "no_head_cover":
                                        cv2.rectangle(new_frame, (int(x3), int(y3)), (int(x4), int(y4)), (0, 0, 255), 2)
                                        cv2.putText(new_frame, f"{label} ({confidence:.2f})", (int(x3), int(y3) - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                    else:
                                        cv2.rectangle(new_frame, (int(x3), int(y3)), (int(x4), int(y4)), (0, 255, 0), 2)
                                        cv2.putText(new_frame, f"{label} ({confidence:.2f})", (int(x3), int(y3) - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                else:
                                    if label in ["BP_measured", "Hand_Sanitized"]:
                                        cv2.rectangle(new_frame, (int(x3), int(y3)), (int(x4), int(y4)), (0, 255, 0), 2)
                                        cv2.putText(new_frame, f"{label} ({confidence:.2f})", (int(x3), int(y3) - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                    else:
                                        cv2.rectangle(new_frame, (int(x3), int(y3)), (int(x4), int(y4)), (0, 0, 255), 2)
                                        cv2.putText(new_frame, f"{label} ({confidence:.2f})", (int(x3), int(y3) - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


                    ret, buffer = cv2.imencode('.jpg', new_frame)
                    new_frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + new_frame + b'\r\n')

                    move_and_zoom_camera(ptz_service, profile, original_pan, original_tilt, 0)
                    t1 = time.time()
                    t2 = time.time()
                    while t2 - t1 < 8:
                        if action == 'start':
                            break
                        success, new_frame = camera.read()
                        if not success:
                            print("Failed to grab frame")
                            break
                        # print(frame.size)

                        #img = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
                        results = model(new_frame, conf=0.1)
                        new_bounding_boxes1 = results[0].boxes.data.to('cpu').numpy()
                        new_bounding_boxes2 = []
                        for box in new_bounding_boxes1:
                            x3, y3, x4, y4, confidence, class_id = box[:6]
                            class_id = int(class_id)
                            if class_id in [0, 3] and confidence < .30:
                                continue
                            new_bounding_boxes2.append([x3, y3, x4, y4, confidence, class_id])
                        # print(new_bounding_boxes)
                        class_ids = [box[-1] for box in new_bounding_boxes2]
                        labels2 = [CLASS_NAMES[i] for i in class_ids]
                        timer2 = time.time()
                        if 'no_head_cover' in labels2 and timer2 - cover_timer > 8:
                            socketio.emit('notification',
                                          {'message': f'Visitor without head cover at {datetime.now().time().strftime("%H:%M:%S")}!'})
                            log_action('Visitor was not wearing headcover.')
                            cover_timer = time.time()
                        if 'no_mask' in labels2 and timer2 - mask_timer > 8:
                            socketio.emit('notification',
                                          {'message': f'Visitor without mask at {datetime.now().time().strftime("%H:%M:%S")}!'})
                            log_action('Visitor was not wearing mask.')
                            mask_timer = time.time()
                        for box in new_bounding_boxes2:
                            x3, y3, x4, y4 = box[:4]  # Convert coordinates to integers
                            confidence = box[4]
                            class_id = int(box[5])
                            label = CLASS_NAMES.get(class_id, "Unknown")
                            if label not in screens:
                                if label == 'Face' and privacy_mode:
                                    try:
                                        roi = new_frame[int(y3) - 20:int(y4) + 3, int(x3) - 20:20 + int(x4)]
                                        roi = cv2.GaussianBlur(roi, (99, 99), 70)
                                        new_frame[int(y3) - 20:int(y4) + 3, int(x3) - 20:20 + int(x4)] = roi
                                    except cv2.error:
                                        roi = new_frame[int(y3):int(y4), int(x3):int(x4)]
                                        roi = cv2.GaussianBlur(roi, (99, 99), 70)
                                        new_frame[int(y3):int(y4), int(x3):int(x4)] = roi
                                elif label == 'Patient_face' and confidence >.4 and patient_privacy_mode:
                                    try:
                                        roi = new_frame[int(y3) - 20:int(y4) + 3, int(x3) - 20:20 + int(x4)]
                                        roi = cv2.GaussianBlur(roi, (99, 99), 70)
                                        new_frame[int(y3) - 20:int(y4) + 3, int(x3) - 20:20 + int(x4)] = roi
                                    except cv2.error:
                                        roi = new_frame[int(y3):int(y4), int(x3):int(x4)]
                                        roi = cv2.GaussianBlur(roi, (99, 99), 70)
                                        new_frame[int(y3):int(y4), int(x3):int(x4)] = roi
                                else:
                                    if 'Apron' in labels2:
                                        if label == "no_head_cover":
                                            cv2.rectangle(new_frame, (int(x3), int(y3)), (int(x4), int(y4)),
                                                          (0, 0, 255), 2)
                                            cv2.putText(new_frame, f"{label} ({confidence:.2f})",
                                                        (int(x3), int(y3) - 10),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                        else:
                                            cv2.rectangle(new_frame, (int(x3), int(y3)), (int(x4), int(y4)),
                                                          (0, 255, 0), 2)
                                            cv2.putText(new_frame, f"{label} ({confidence:.2f})",
                                                        (int(x3), int(y3) - 10),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                    else:
                                        if label in ["BP_measured", "Hand_Sanitized"]:
                                            cv2.rectangle(new_frame, (int(x3), int(y3)), (int(x4), int(y4)),
                                                          (0, 255, 0), 2)
                                            cv2.putText(new_frame, f"{label} ({confidence:.2f})",
                                                        (int(x3), int(y3) - 10),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                        else:
                                            cv2.rectangle(new_frame, (int(x3), int(y3)), (int(x4), int(y4)),
                                                          (0, 0, 255), 2)
                                            cv2.putText(new_frame, f"{label} ({confidence:.2f})",
                                                        (int(x3), int(y3) - 10),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            elif label in screens and confidence >.8:
                                if label in ['Paramonitor', 'Ventilator']:
                                    cv2.rectangle(new_frame, (int(x3), int(y3)), (int(x4), int(y4)), (0, 255, 0), 2)
                                    cv2.putText(new_frame, f"{label} ({confidence:.2f})", (int(x3), int(y3) - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                else:
                                    cv2.rectangle(new_frame, (int(x3), int(y3)), (int(x4), int(y4)), (0, 0, 255), 2)
                                    cv2.putText(new_frame, f"{label} ({confidence:.2f})", (int(x3), int(y3) - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        ret, buffer = cv2.imencode('.jpg', new_frame)
                        new_frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + new_frame + b'\r\n')
                        t2 = time.time()

                # elif label not in screens:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label_main} ({confidence_main:.2f})", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                elif label_main not in screens:

                    if label_main != 'Face' and label_main != 'Patient_face':
                        if 'Human' not in labels1:
                            ground1 = 0
                        if 'Human' in labels1 and "Apron" not in labels1:
                            now = datetime.now().time()
                            time_stamp = now.strftime("%H:%M:%S")
                            detected1.insert(0, time_stamp)
                            detected1.append('Human')

                            if ground1 == 0:
                                socketio.emit('notification', {'message': 'Person without Apron detected!'})
                                log_action('Person without Apron detected')
                                ground1 = 1
                            if label_main == 'Human':
                                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                                cv2.putText(frame, f"{label_main} ({confidence_main:.2f})", (int(x1), int(y1) - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            elif label_main == 'no_head_cover':
                                detected1.append(label_main)
                                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                                cv2.putText(frame, f"{label_main} ({confidence_main:.2f})", (int(x1), int(y1) - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, (0, 0, 255), 2)
                            else:
                                detected1.append(label_main)
                                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                                cv2.putText(frame, f"{label_main} ({confidence_main:.2f})", (int(x1), int(y1) - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, (0, 255, 0), 2)
                        else:
                            if label_main == 'Human':
                                now = datetime.now().time()
                                time_stamp = now.strftime("%H:%M:%S")
                                detected1.insert(0, time_stamp)
                                detected1.append('Human')
                            if label_main == 'no_head_cover':
                                detected1.append(label_main)
                                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                                cv2.putText(frame, f"{label_main} ({confidence_main:.2f})", (int(x1), int(y1) - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, (0, 0, 255), 2)
                            else:
                                detected1.append(label_main)
                                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                                cv2.putText(frame, f"{label_main} ({confidence_main:.2f})", (int(x1), int(y1) - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, (0, 255, 0), 2)
                    elif label_main == 'Face' and privacy_mode:
                        try:
                            roi = frame[int(y1) - 20:int(y2) + 3, int(x1) - 20:20 + int(x2)]
                            roi = cv2.GaussianBlur(roi, (99, 99), 70)
                            frame[int(y1) - 20:int(y2) + 3, int(x1) - 20:20 + int(x2)] = roi
                        except cv2.error:
                            roi = frame[int(y1):int(y2), int(x1):int(x2)]
                            roi = cv2.GaussianBlur(roi, (99, 99), 70)
                            frame[int(y1):int(y2), int(x1):int(x2)] = roi
                    elif label_main == 'Patient_face' and confidence_main > .4 and patient_privacy_mode:
                        try:
                            roi = frame[int(y1) - 20:int(y2) + 3, int(x1) - 20:20 + int(x2)]
                            # print("here1")
                            roi = cv2.GaussianBlur(roi, (99, 99), 70)
                            frame[int(y1) - 20:int(y2) + 3, int(x1) - 20:20 + int(x2)] = roi
                        except cv2.error:
                            roi = frame[int(y1):int(y2), int(x1):int(x1)]
                            # print("Here2")
                            roi = cv2.GaussianBlur(roi, (99, 99), 70)
                            frame[int(y1):int(y2), int(x1):int(x2)] = roi
                        finally:
                            continue

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            start_time2 = time.time()
            if monitor_time <= start_time2-start_time1 and action in ['start', 'hold']:
                action = "cancel"
                socketio.emit('notification', {'message': f'Monitoring duration complete. Camera is in rotation mode now!'})
                print("Monitoring complete. Camera is back on rotation mode. ")
                break

        if action == "cancel":
            if 0 <= pan < 1:
                pan += .42
                move_and_zoom_camera(ptz_service, profile, pan, tilt, zoom)
            elif pan >= 1:
                pan = -0.195
                move_and_zoom_camera(ptz_service, profile, pan, tilt, zoom)
            elif 0 > pan > -1:
                pan -= 0.39
                move_and_zoom_camera(ptz_service, profile, pan, tilt, zoom)
            elif pan <= -1:
                rotation += 1
                pan = 0
                print("! Rotation completed")
                move_and_zoom_camera(ptz_service, profile, pan, tilt, zoom)


@app.route('/')
def index():
    return render_template('AIIMS.html')


@app.route("/submit", methods=["POST"])
def submit():
    global bed_no
    global action
    global monitor_till

    data = request.get_json()
    bed_number = int(data.get("beds"))
    monitor_till = data.get("appt")
    action = data.get("action")  # 'start' or 'cancel'
    print(action)
    print(f"Action: {action.upper()}, Bed: {bed_number}, Monitor till: {monitor_till}")

    seconds_to_monitor = 0
    if action == "start":
        bed_no = bed_number
        seconds_to_monitor = seconds_until(monitor_till)
    elif action == "cancel":
        # Cancel monitoring logic
        pass

    return jsonify({
        "status": "success",
        "message": f"{action.capitalize()} received!",
        "seconds_left": seconds_to_monitor
    })


def add_overlay(frame, text):
    """Adds DOS-style text overlay with a rectangular box around it."""

    font = cv2.FONT_HERSHEY_PLAIN  # Pixelated DOS-style font
    font_scale = 2
    font_color = (255, 255, 255)  # Green like classic DOS text
    thickness = 2
    position = (50, 50)

    # Get text size for dynamic box dimensions
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_w, text_h = text_size[0], text_size[1]
    box_padding = 10  # Padding around the text
    box_tl = (position[0] - box_padding, (position[1] - box_padding) - 20)  # Top-left corner
    box_br = ((position[0] + text_w + box_padding), (position[1] + text_h + box_padding) - 14)  # Bottom-right corner

    # Draw only the outline of the rectangular box (no fill)
    box_thickness = 2  # Adjust thickness of the rectangle outline
    cv2.rectangle(frame, box_tl, box_br, (255, 255, 255), box_thickness)
    # Draw text shadow for retro look
    cv2.putText(frame, text, (position[0] + 2, position[1] + 2), font, font_scale, (0, 0, 0), thickness + 1,
                cv2.LINE_AA)

    # Draw text
    cv2.putText(frame, text, position, font, font_scale, font_color, thickness, cv2.LINE_AA)

    return frame


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_daily_logs')
def get_daily_logs():
    data_path = os.path.join(excel_output_dir, 'logs.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        return df.tail(5).to_json(orient='records')  # Serve the latest 10 rows
    return []


@app.route('/get_ventilator_data')
def get_ventilator_data():
    data_path = os.path.join(excel_output_dir, 'ventilator.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        return df.tail(10).to_json(orient='records')  # Serve the latest 10 rows
    return []


@app.route('/get_paramonitor_data')
def get_paramonitor_data():
    data_path = os.path.join(excel_output_dir, 'paramonitor.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        return df.tail(10).to_json(orient='records')  # Serve the latest 10 rows
    return []


##############################################################################################################

##############################################################################################################


# Global frame storage

def human_tracking():
    # camera1 = cv2.VideoCapture(1)
    global logs
    global frame_1
    global previous_count
    global previous_frame
    run = 0
    while True:
        try:
            frame = frame_queue.get(timeout=1)
            # print("Debug_point")
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame, conf=0.1)
            bounding_boxes = results[0].boxes.data.to('cpu').numpy()
            # print(bounding_boxes)
            new_bounding_boxes = []
            for box in bounding_boxes:
                x1, y1, x2, y2, confidence, class_id = box[:6]
                class_id = int(class_id)
                if class_id == 6 and confidence < .60:
                    continue
                else:
                    timer_for_last_detection = time.time()
                new_bounding_boxes.append([x1, y1, x2, y2, confidence, class_id])
            class_ids = [box[-1] for box in new_bounding_boxes]
            labels1 = [CLASS_NAMES[i] for i in class_ids]
            if 'Human' in labels1 and previous_frame == False:
                if time.time() - timer_for_last_detection > 5:
                    events = Counter(labels1)
                    counts = events['Human']
                    logs['No_of_person'] += counts
                    previous_count = counts
                    previous_frame = True
                if run == 0:
                    events = Counter(labels1)
                    counts = events['Human']
                    logs['No_of_person'] += counts
                    previous_count = counts
                    previous_frame = True
                    run = 1
            elif 'Human' in labels1 and previous_frame == True:
                events = Counter(labels1)
                counts = events['Human']
                if counts > previous_count:
                    new_count = counts - previous_count
                    logs['No_of_person'] += new_count
                previous_count = counts
                previous_frame = True
            elif 'Human' not in labels:
                previous_frame = False
                previous_count = 0
            print(logs)
        except queue.Empty:
            pass


def save_data():
    loging = 0
    while True:
        try:
            # with open(r'C:\Users\OMEN\PycharmProjects\object_detection\yolov7\excel_output\data.txt', 'r') as file:
            #     lines = file.readlines()
            #     if len(lines) >= 2:  # ✅ Ensure the file has at least two lines
            #         current_date = lines[0].strip()  # ✅ Read the first line (Date)
            #         current_time = lines[1].strip()  # ✅ Read the second line (Time)
            #         application = lines[2].strip()
            #     else:
            #         current_date, current_time = None, None

            data, current_date, current_time, application_name = process_yolo_images()
            # print(f'This is {current_date}, {current_time}')  # ✅ Now, always prints correct values
            if current_date:
                save_data_to_excel(data, current_date, current_time, application_name)
            time.sleep(2)

        except:
            continue


if __name__ == "__main__":
    if not camera.isOpened():
        print("Error: Unable to open camera.")
    else:
        threading.Thread(target=human_tracking, daemon=True).start()
        # if not any(t.name == "save_data_thread" for t in threading.enumerate()):
        #     threading.Thread(target=save_data, name="save_data_thread", daemon=True).start()
        threading.Thread(target=check_midnight, daemon=True).start()
        socketio.run(app, debug=True)

