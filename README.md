# **Image Story Teller 🤖**
📖 This is an multimodal AI application for story generation, allowing you to generate a vivid story audio from a picture seamlessly!

![Hugging Face Spaces](https://img.shields.io/badge/Deployed-Hugging%20Face-blue)  
![Python](https://img.shields.io/badge/Python-3.9-blue.svg)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange)

---

## 🎯 **Project Overview**
Image Story Teller is a multimodal AI application powered by **DeepSeek-R1-Distill-Qwen-1.5B**, **blip-image-captioning-large** and **Kokoro-82M**, allowing users to generate a short story based on the input image using AI. 
This project utilizes **Gradio** for an intuitive web UI and is deployed on **Hugging Face Spaces** for easy access.

🔗 **Live Demo: [Try it on Hugging Face](https://huggingface.co/spaces/DaisioSK/image-story-teller)**

---

## 🛠 How It Works  
**Image Story Teller** follows an intelligent three-step process to generate immersive audio stories from images:  

### 1️⃣ Image to Caption  
The AI analyzes the uploaded image and generates a concise, meaningful caption summarizing its content.  

### 2️⃣ Caption to Story  
The caption is then fed into **DeepSeek** LLM model, which expands it into a rich, engaging story with vivid details and natural language fluency.  

### 3️⃣ Story to Speech  
Finally, the generated story is converted into high-quality speech, allowing users to download and listen to the AI-narrated story.  

---

## 🚀 **Features**
- ✅ **AI Story Generation**: Expands your story based on the given image using DeepSeek.
- ✅ **User-Friendly Interface**: Built with Gradio, making it easy to use.
- ✅ **Deployed on Hugging Face**: No need for local installation, simply access online.
- ✅ **Store Conversations Locally**: modify config.py to enable conversation logging.
  
---

## 📥 **Installation & Usage**
To run this project locally, follow these steps:

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/DaisioSK/Image-Story-Teller.git
cd Image-Story-Teller
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Run the Application**
```bash
python app.py
```
Once started, visit **`http://127.0.0.1:7860/`** in your browser to interact with the Image Story Teller.

---


## 📜 **License**
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🎉 **Thank You!**
If you find this project useful, consider giving it a **⭐ Star**! 🚀✨

