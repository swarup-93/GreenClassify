# 🥬 GreenClassify  
## Intelligent Vegetable Image Classification Using Deep Learning

---

## 📌 Overview

GreenClassify is a deep learning–powered web application designed to classify vegetable images into predefined categories.
The system utilizes **Convolutional Neural Networks (CNNs)** for image recognition and is deployed using the **Flask web framework**.
Users can upload vegetable images through a browser interface and receive accurate predictions in real time.

This project demonstrates a complete end-to-end machine learning workflow, from data preparation and model training to deployment and user interaction.

---

## ❓ Problem Description

Manual classification of vegetables in agricultural processing units and retail stores is inefficient and error-prone.
Large volumes of vegetables require fast and consistent identification to support sorting, quality control, and inventory management.

Traditional methods struggle with:
- Human errors  
- Slow processing speed  
- Inconsistent classification  
- Increased operational cost  

GreenClassify addresses these challenges by automating vegetable identification using deep learning.

---

## 🎯 Objectives

- Understand and implement CNN-based image classification  
- Preprocess and augment image datasets for better learning  
- Train and evaluate a deep learning model  
- Deploy the trained model using a Flask web application  
- Provide a simple and interactive user interface  

---

## 🧠 System Approach

The system follows the workflow below:

1. Image dataset collection and organization  
2. Data preprocessing and augmentation  
3. CNN model construction and training  
4. Model evaluation using test data  
5. Model serialization for deployment  
6. Integration with Flask backend  
7. Real-time prediction through web interface  

---

## 🛠️ Technology Stack

- **Language:** Python  
- **Deep Learning:** TensorFlow, Keras  
- **Web Framework:** Flask  
- **Frontend:** HTML, CSS, JavaScript  
- **Development Tools:** Google Colab, VS Code  
- **Dataset Source:** Kaggle  

---

## 📂 Project Structure

```
vegetable_flask_app/
│
├── app.py
├── vegetable_classifier.h5
├── class_map.pkl
├── requirements.txt
│
├── templates/
│   ├── index.html
│   ├── prediction.html
│   └── logout.html
│
├── static/
│   ├── css/main.css
│   └── js/main.js
│
├── uploads/
└── screenshots/
```

---

## ⚙️ Application Workflow

1. User accesses the home page  
2. Image is uploaded  
3. Image is resized and normalized  
4. CNN model predicts the vegetable class  
5. Result is displayed to the user  

---

## 📊 Results

The trained CNN model accurately classifies vegetables such as Broccoli, Tomato, Carrot, and Brinjal.
Testing confirms strong generalization on unseen data and reliable real-time predictions via the Flask interface.

---

## 🔮 Future Scope

- Implement transfer learning (MobileNet, ResNet)  
- Deploy on cloud platforms  
- Extend classification to fruits  
- Integrate real-time camera input  

---

## 📚 References

- TensorFlow Documentation  
- Keras Documentation  
- Flask Official Documentation  
- Kaggle Datasets  
- W3Schools  

---

© 2026 GreenClassify Project
