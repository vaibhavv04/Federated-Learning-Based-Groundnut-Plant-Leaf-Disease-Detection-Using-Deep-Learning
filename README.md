# 🌿 Federated Learning-Based Groundnut Leaf Disease Detection

This project aims to **detect diseases in groundnut plant leaves** using deep learning and **federated learning** techniques, enabling scalable, privacy-preserving model training across decentralized clients.

---

## 🚀 Objective

To develop a **Federated Learning system** using **TensorFlow Federated** and the **FedAvg algorithm** to identify groundnut leaf diseases such as:

- Early Leaf Spot  
- Late Leaf Spot  
- Rust  
- Nutrient Deficiency  

**Key goals:**
- Enhance model generalizability using diverse annotated datasets
- Minimize communication overhead
- Support real-time insights for farmers and agricultural stakeholders

---

## 📚 Dataset

- **Kaggle Groundnut Leaf Dataset**
- **Mendeley Groundnut Leaf Dataset**

To balance client datasets:
- Client 1 & 2 used splits from the Kaggle dataset
- Client 3 used the Mendeley dataset, upsampled using augmentation:
  - Random rotations and flips  
  - Gaussian noise, blur  
  - Shift, scale, and rotate  
  - Hue, saturation, and value changes  

---

## 🧪 Models Used

- `MobileNet` – Efficient and accurate
- `NasNet Mobile` – Lightweight, mobile-friendly
- `DenseNet121` – Strong feature reuse and gradient flow

---

## 🏗️ System Design

- 3 Federated Clients:
  - **Client 1 & 2**: Kaggle dataset (split)
  - **Client 3**: Mendeley dataset (augmented)
- Federated averaging (`FedAvg`) for model aggregation
- Preprocessing and augmentation pipeline automated

---

## ⚙️ Tools & Libraries

- Python  
- TensorFlow Federated  
- NumPy  
- OpenCV  
- Matplotlib  
- Jupyter Notebooks / Python scripts  

---

## 📈 Future Work

- Add real-world groundnut leaf images from fields
- Experiment with advanced deep learning architectures
- Integrate better augmentation pipelines for minority classes



---

> ⚡ This project supports privacy-friendly and distributed machine learning in the agriculture domain.
