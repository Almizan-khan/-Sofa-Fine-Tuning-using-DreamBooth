# -Sofa-Fine-Tuning-using-DreamBooth
This project fine-tunes the `runwayml/stable-diffusion-v1-5` model on custom sofa images using the **DreamBooth**
It helps generate realistic sofa designs and variations (e.g., color, lighting, style) using your own dataset.

---

## 🚀 Project Overview
The `sofa.py` script automates the entire training workflow:
1. Installs all required dependencies.
2. Connects to Google Drive for dataset storage.
3. Uploads and organizes training images.
4. Fine-tunes the Stable Diffusion model with specified prompts and parameters.
5. Saves the output model in a new directory for later inference.

---

## 🧠 Technologies Used
- **Python 3.10+**
- **PyTorch**
- **Hugging Face Diffusers**
- **Transformers**
- **PEFT (Parameter-Efficient Fine-Tuning)**
- **Accelerate**
- **xFormers**
- **BitsAndBytes**

---

## 📦 Installation

Run this script in **Google Colab** or a local environment with GPU.


