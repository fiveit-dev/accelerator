# Alquimia AI Accelerator

Welcome to the **Alquimia AI Accelerator** repository! This "hands-on" repository is designed as a quick-start guide to help you understand and implement Alquimia AI capsules across various domains, including NLP, computer vision, and audio.

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
