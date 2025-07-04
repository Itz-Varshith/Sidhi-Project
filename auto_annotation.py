import os
import time
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
import re 
from PIL import Image
import shutil
import threading


def check_pair(file_list, file_name):
    base_name, ext = os.path.splitext(file_name)
    if ext.lower() == ".jpg":
        required = base_name + ".txt"
    elif ext.lower() == ".txt":
        required = base_name + ".jpg"
    else:
        return False  # skip unknown file types

    return required in file_list

def delete_file(frame_dir, image, txt):
    image_path = os.path.join(frame_dir, image)
    txt_path = os.path.join(frame_dir, txt)
    print("Removing:", image_path, txt_path)
    try:
        if os.path.exists(image_path):
            os.remove(image_path)
        if os.path.exists(txt_path):
            os.remove(txt_path)
    except Exception as e:
        print(f"Error deleting files: {e}")

def load_model():
    model = AutoModelForImageTextToText.from_pretrained("qwen2.5-vl-7b-instruct-quantized",device_map = "auto")
    processor = AutoProcessor.from_pretrained("qwen2.5-vl-7b-instruct-quantized")
    return model,processor

def generate_bbox(image_path, object_class, model, processor):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant that gives YOLO base bounding box."
                "Only return the values in [x1, y1, x2, y2] format."
                "Do not give out labels."
                "Output format:\n"
                "bbox: <value or None>\n\n"
                "Do not include any other text, no greetings, and no explanations. "
                "Output must begin directly with 'bbox:'."
            )
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": f"From the image, give me bounding box of the {object_class} present in the image. And if {object_class} is not present, give none."}
            ]
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print(f"Model Output: {output_text}")

    if "None" in output_text:
        return None

    match = re.search(r'bbox:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', output_text)
    if match:
        return list(map(int, match.groups()))
    return None

def convert_to_yolov8(bbox, img_width, img_height):
    x_center = (bbox[0] + bbox[2]) / 2 / img_width
    y_center = (bbox[1] + bbox[3]) / 2 / img_height
    width = (bbox[2] - bbox[0]) / img_width
    height = (bbox[3] - bbox[1]) / img_height
    return f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def main(frame_dir,image,txt,model,processor):
    # pass the input to the model to get bbox
    des_folder = os.path.join("Sidhi-Project","backend","public","unverified")
    base, ext = os.path.splitext(image)
    image_path = os.path.join(frame_dir,image)
    txt_path = os.path.join(frame_dir,txt)
    with open(txt_path) as f:
        object_class = f.read().strip()
     
    output = generate_bbox(image_path=image_path,object_class=object_class,model=model,processor=processor)
    if output:
        # img = Image.open(image_path)
        # img_width = img.width
        # img_height = img.height
        with Image.open(image_path) as img:
            img_width = img.width
            img_height = img.height
        # change it to yolov8 formate
        convert_yolo_formate = convert_to_yolov8(output,img_height=img_height,img_width=img_width)
        # save the image,prompt, box
        # 1st image 
        des_path = os.path.join(des_folder,image)
        shutil.move(image_path,des_path)
        # prompt 
        prompt_path = base + "_prompt.txt"
        des_prompt = os.path.join(des_folder,prompt_path)
        with open(des_prompt,"w") as f:
            f.write(object_class)
        # bbox
        bbox = base + "_bbox.txt"
        des_bbox = os.path.join(des_folder,bbox)
        with open(des_bbox,"w") as f:
            f.write(convert_yolo_formate)
    # delelte teh file
    delete_file(frame_dir, base + ".jpg", base + ".txt")


def monitor_dir():
    frame_dir = os.path.join("Sidhi-Project", "backend", "routes", "frames")

    seen_files = set()
    # laod the model
    model,processor = load_model()
    while True:
        try:
            current_files = set(os.listdir(frame_dir))
            new_files = current_files - seen_files
            seen_files.update(new_files)

            for file in list(new_files):
                if file.endswith((".jpg", ".txt")):
                    base, ext = os.path.splitext(file)
                    pair_file = base + (".txt" if ext == ".jpg" else ".jpg")

                    if pair_file in current_files:
                        print(f"Pair found: {file}, {pair_file}")
                        main(frame_dir, base + ".jpg", base + ".txt",model,processor)
                        # Remove from seen_files so it's not double-processed
                        seen_files.discard(base + ".jpg")
                        seen_files.discard(base + ".txt")

        except KeyboardInterrupt:
            print("Stopped by user")
            break
        except Exception as e:
            print(f"Error in loop: {e}")
            time.sleep(5)

if __name__ == "__main__":
    thread = threading.Thread(target=monitor_dir, daemon=True)
    thread.start()

    print("Monitoring thread started. Press Ctrl+C to exit.")
    
    try:
        while True:
            time.sleep(1)  # Main thread stays alive
    except KeyboardInterrupt:
        print("Shutting down...")
