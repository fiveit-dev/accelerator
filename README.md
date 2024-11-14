# Alquimia AI Accelerator

Welcome to the **Alquimia AI Accelerator** repository! This "hands-on" repository is designed as a quick-start guide to help you understand and implement Alquimia AI capsules across various domains, including NLP, computer vision, and audio.

## Repository Structure

The accelerator is divided into three main sections, each containing a variety of ready-to-use implementations and examples in the following fields:

1. **Natural Language Processing (NLP)**

   - NLP, or Natural Language Processing, is a branch of AI focused on enabling computers to understand, interpret, and generate human language. It involves techniques and algorithms that allow machines to process and analyze large amounts of natural language data.
   - This section provides capsules that demonstrate NLP implementations, ranging from text processing and sentiment analysis to advanced language generation.
   - [NLP Repository Link](https://github.com/Alquimia-ai/nlp.git)

2. **Computer Vision**

   - The computer vision section includes capsules dedicated to enabling machines to interpret and make decisions based on visual inputs, such as images or video. Use cases range from object detection and facial recognition to scene segmentation.
   - [Computer Vision Repository Link](https://github.com/Alquimia-ai/computer-vision.git)

3. **Audio Processing**
   - This section focuses on audio processing techniques, such as speech recognition, sound classification, and audio signal analysis. These capsules demonstrate various methods for working with and extracting insights from audio data.
   - [Audio Repository Link](https://github.com/Alquimia-ai/audio.git)

## Prebuilt Capsule for OpenshiftAI, Label Studio, and MLFlow

This repository also includes a prebuilt capsule that is ready to use with **OpenshiftAI**, **Label Studio**, and **MLFlow**. This capsule is designed to enhance your workflow by integrating core AI tools for streamlined development and deployment.

### Getting Started Guide

To get started with the prebuilt spam filter capsule, follow these steps:

1. Navigate to the `capsules/spam-filter` directory.
2. Install the required packages using the `requirements.txt` file:

   ```bash
   pip install -r capsules/spam-filter/requirements.txt
   ```

3. Run the main Python script to launch the capsule:

```bash
python capsules/spam-filter/main.py
```

### Requirements for the Spam Filter Capsule

Ensure that the following tools and services are available:

- **MLFlow**: For experiment tracking and model management.
- **Label Studio**: For data labeling and annotation.
- **Openshift AI**: A scalable AI infrastructure platform.
- **Alquimia AI Custom Triton Inference Server**: This server must be added to Openshift AI for full integration and deployment.

---

Explore each section, dive into the code, and experiment with the capsules to enhance your understanding of AI applications in NLP, computer vision, and audio. Enjoy your journey with Alquimia AI!
