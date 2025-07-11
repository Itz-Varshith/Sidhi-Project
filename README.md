# React + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [`typescript-eslint`](https://typescript-eslint.io) in your project.
## System Requirments
### Hardware Requirments
OS: Ubuntu 20.04+ / Windows 10+


GPU: NVIDIA GPU with minimum 12 GB VRAM (e.g., RTX 3080/3090, A100 preferred)


RAM: 16 GB minimum


Disk: 10+ GB free space


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
pip install opencv-python
pip install flask
pip install transformers
pip install accelerate
pip install Pillow
```


