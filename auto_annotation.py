from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
import torch
import bitsandbytes
import re
import os
from qwen_vl_utils import process_vision_info
from PIL import Image
import shutil

def load_model():
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    model = AutoModelForImageTextToText.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        quantization_config=bnb_config,
        device_map="auto"
    )
    return model, processor

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
        videos=video_inputs,
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
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

if __name__ == "__main__":
    model, processor = load_model()

    FRAMES_DIR = os.path.join('backend', 'routes', 'frames')
    UNVERIFIED_DIR = os.path.join('public', 'unverified')

    os.makedirs(UNVERIFIED_DIR, exist_ok=True)

    if not os.path.exists(FRAMES_DIR):
        print(f"Directory not found: {FRAMES_DIR}")
        exit(1)

    files = os.listdir(FRAMES_DIR)
    image_files = [f for f in files if f.endswith('.jpg')]

    for image_file in image_files:
        base_name = os.path.splitext(image_file)[0]
        txt_file = f"{base_name}.txt"
        txt_path = os.path.join(FRAMES_DIR, txt_file)
        image_path = os.path.join(FRAMES_DIR, image_file)

        if not os.path.exists(txt_path):
            print(f"Skipping {image_file}: prompt file not found.")
            continue

        with open(txt_path, 'r') as f:
            prompt = f.read().strip()

        print(f"Processing: {image_file} with prompt: {prompt}")

        bbox = generate_bbox(image_path, prompt, model, processor)
        if bbox is None:
            print(f"No bbox found for {image_file}. Skipping.")
            continue

        # Load image size
        img = Image.open(image_path)
        yolov8_txt = convert_to_yolov8(bbox, img.width, img.height)

        # Define save paths
        save_image_path = os.path.join(UNVERIFIED_DIR, image_file)
        save_prompt_path = os.path.join(UNVERIFIED_DIR, f"{base_name}_prompt.txt")
        save_bbox_path = os.path.join(UNVERIFIED_DIR, f"{base_name}_bbox.txt")

        # Copy image and save annotation files
        shutil.copy(image_path, save_image_path)
        with open(save_prompt_path, 'w') as f:
            f.write(prompt)
        with open(save_bbox_path, 'w') as f:
            f.write(yolov8_txt)

        print(f"✅ Saved to {UNVERIFIED_DIR}: {base_name}.jpg, _prompt.txt, _bbox.txt")

