# Sidhi ICU Event tracking Project
This project is an end-to-end system for automatic image annotation and incremental training of a YOLOv8 model, tailored specifically for real-time monitoring of ICU events. The goal is to assist doctors and medical staff by providing intelligent visual tracking and alert capabilities through a web-based interface.

Key features of the project:

 - Live Frame Capture: Continuously captures frames from an MJPEG stream.

 - AI-Powered Annotation: Uses a multimodal Vision-Language Model (Qwen-VL) to auto-detect and localize critical objects or people (e.g., patient, nurse, doctor) from user prompts.

 - YOLOv8-Compatible Labeling: Bounding box outputs are formatted directly into YOLOv8-style annotations for seamless training.

 - User-in-the-Loop Verification: Doctors or annotators can review and verify predicted labels via a web dashboard before saving.

 - Incremental Learning: Once enough verified samples are collected, the backend automatically triggers a retraining of the YOLOv8 model to improve future detections.

 - Web Interface: Built with Node.js and Express, the system provides a clean, user-friendly UI to manage captured data, verify outputs, and track model improvement over time.

This system is designed for scalability, speed, and usability in medical environments, allowing healthcare professionals to quickly build and adapt computer vision systems without needing deep technical knowledge.

## System Requirments
### Hardware Requirments

 - OS: Ubuntu 20.04+ / Windows 10+


 - GPU: NVIDIA GPU with minimum 12 GB VRAM (e.g., RTX 3080/3090, A100 preferred)


 - RAM: 16 GB minimum


 - Disk: 10+ GB free space


### Software Requirements 
| Tool    | Version                                   |
| ------- | ----------------------------------------- |
| Node.js | 18.x or 20.x                              |
| Python  | 3.9 or 3.10                               |
| CUDA    | 11.8 or 12.x                              |
| PyTorch | 2.0+                                      |
| pip     | Latest                                    |

## Instruction to Setup 

1. Create Python virtual environment
```bash
python -m venv sidhi
source sidhi/bin/activate  # Linux/macOS
sidhi\Scripts\activate     # Windows
```
2. Install Dependencis
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 # Install torch according to your cuda
pip install qwen_vl_utils
pip install opencv-python
pip install flask
pip install transformers
pip install accelerate
pip install Pillow
pip install ultralytics
```
3. Clone the repository
```bash
git clone https://github.com/Itz-Varshith/Sidhi-Project.git
cd Sidhi-Project
```
4. Open the terminal and run the backend python file
```bash
python main2_new.py
```
5. Open the new terminal and the run the js node for forntend
```bash
cd src
npm run dev
cd ..
```
6. Open a new terminal to run main Router file
```bash
cd backend
node index.js
cd ..
```
7. Open a new terminal run the auto annotation script
```bash
python auto_annotation.py
```






