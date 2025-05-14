# ml-deployment-playground
Train a deep learning model on the Digits dataset and deploy it across various platforms including Flask, FastAPI, Streamlit, Docker, AWS, Heroku, TensorFlow Serving, and more.

This repository provides an end-to-end pipeline for building and deploying a deep learning model using the popular digits dataset from scikit-learn. It covers model training, evaluation, serialization, and deployment to multiple environments and platforms. Whether you're deploying locally with Flask or Streamlit, using containers via Docker, or scaling in the cloud with AWS, GCP, or Azure, this project serves as a one-stop reference for production-ready machine learning deployment workflows.

## 🔍 Overview

This repository showcases:

- ✅ Data preparation and training using Keras (TensorFlow)
- ✅ Evaluation and serialization of the model
- ✅ REST API deployment using Flask and FastAPI
- ✅ UI-based deployment with Streamlit and Gradio
- ✅ Containerized deployment via Docker
- ✅ Cloud deployment on AWS SageMaker, Google Cloud AI, Azure ML
- ✅ Hosting via Heroku and Hugging Face Spaces
- ✅ Model serving using TensorFlow Serving and ONNX Runtime

---

## 📦 Folder Structure

```plaintext
ml-deployment-playground/
│
├── model/                      # Trained model files
│   ├── digits_model.h5
│   └── class_names.pkl
│
├── training/                   # Model training scripts
│   └── train_model.py
│
├── flask_app/                  # Flask deployment
│   └── app.py
│
├── streamlit_app/             # Streamlit deployment
│   └── app.py
│
├── fastapi_app/               # FastAPI deployment
│   └── main.py
│
├── docker/                    # Docker deployment files
│   ├── Dockerfile
│   └── app.py
│
├── gradio_app/                # Gradio interface
│   └── app.py
│
├── tf_serving/                # TensorFlow serving export
│   └── saved_model/...
│
├── onnx/                      # ONNX model export
│   └── model.onnx
│
├── requirements.txt
├── README.md
└── .gitignore
