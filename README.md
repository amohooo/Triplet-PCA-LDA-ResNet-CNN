# 🧍‍♂️ Person Re-Identification Using Classical and Deep Learning Methods

## 📘 Project Overview

This project focuses on **person re-identification**, which aims to match a probe image of a person to their identity in a gallery of known individuals. The task is approached through both **classical machine learning techniques** (PCA + LDA) and **deep metric learning** using **Triplet Networks with a ResNet backbone**. Evaluation includes **Top-N accuracy** and **Cumulative Match Characteristic (CMC) curves**.

---

## 🧠 Objectives

- Load and preprocess a portion of the Market-1501 dataset
- Build a non-deep learning pipeline using PCA and LDA
- Train a deep learning Triplet Network using a ResNet-like CNN
- Evaluate performance with Top-1, Top-5, and Top-10 accuracy
- Generate and visualize CMC curves
- Compare both methods on effectiveness and runtime
- Reflect on ethical implications of person re-identification

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **TensorFlow / Keras** – Deep learning
- **OpenCV** – Image processing
- **scikit-learn** – PCA, LDA, metrics
- **Seaborn / Matplotlib** – Visualization
- **NumPy / SciPy** – Numerical computing
- **Market-1501 Dataset** – Person re-ID dataset (subset)

---

## 🧪 Features Implemented

### 🔹 Data Handling
- Custom loaders for **Training**, **Gallery**, and **Probe** sets
- Tools to:
  - Resize, grayscale, and vectorize images
  - Create Siamese and Triplet batches for metric learning

### 🔹 Classical Approach
- Image vectorization
- **PCA** for dimensionality reduction
- **LDA** for discriminative subspace learning
- Evaluation via Top-N accuracy and CMC curves

### 🔹 Deep Learning Approach
- **Triplet Network** with ResNet-style feature extractor
- Data augmentation pipeline using Keras
- Custom **Triplet Loss Layer**
- Evaluation based on embeddings using L2 distance
- High-quality CMC curve visualizations

---

## 📊 Evaluation Metrics

- **Top-1 Accuracy**: Correct match ranked first
- **Top-5/10 Accuracy**: Correct match in top 5 or 10
- **CMC Curve**: Cumulative accuracy over ranks
- **Confusion Matrix**: For classifier-based models (e.g., Random Forest)
- **Training & Inference Time**: Runtime comparisons

---

## 📈 Example Results

| Method             | Top-1 | Top-5 | Top-10 |
|--------------------|-------|-------|--------|
| PCA + LDA + L1     | 0.54  | 0.71  | 0.80   |
| Triplet Network    | 0.77  | 0.91  | 0.95   |

*Note: Values are illustrative. Please refer to the latest evaluation runs.*

---

## 🔐 Ethical Considerations

- **Privacy**: Re-ID systems can be used in surveillance; ethical concerns around mass data collection.
- **Bias**: Potential for demographic bias depending on dataset diversity.
- **Consent**: Dataset creation must respect subject consent and fair use.
- **Accountability**: Misidentification in real-world scenarios can have serious consequences.

This project considers these by using public datasets and applying the work strictly for educational and experimental purposes.


---
