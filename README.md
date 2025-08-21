# 🌾 Crop Recommendation System using Machine Learning

A smart AI-powered system that recommends the most suitable crop to grow based on soil and environmental conditions. This project leverages supervised machine learning algorithms to improve agricultural productivity and decision-making.

---

## 📌 Project Overview

Choosing the right crop based on soil properties and climatic factors is crucial for maximizing yield. This system uses real-time or static inputs like **Nitrogen (N)**, **Phosphorus (P)**, **Potassium (K)**, **Temperature**, **Humidity**, **pH**, and **Rainfall** to recommend the ideal crop using a trained ML model.

---

## 🚀 Technologies & Tools Used

- Python  
- Pandas, NumPy  
- Scikit-learn (Random Forest, SVM, etc.)  
- Matplotlib, Seaborn
- Flask

---

## 📂 Dataset

- **Source:** Kaggle (Crop Recommendation Dataset)  
- **Features:**  
  - `N`: Nitrogen level in soil  
  - `P`: Phosphorus level  
  - `K`: Potassium level  
  - `temperature`, `humidity`, `ph`, `rainfall`  
- **Target:** Recommended Crop (e.g., rice, maize, cotton)

---

## 📈 Workflow

1. **Data Preprocessing**
   - Handle missing values (if any)
   - Check for data balance
2. **Exploratory Data Analysis**
   - Correlation heatmaps
   - Class distribution
3. **Model Training**
   - Train-Test Split (80-20)
   - Model: Random Forest Classifier (best accuracy)
   - Accuracy: ~89%
4. **Evaluation**
   - Confusion Matrix
   - Classification Report
5. **Prediction**
   - Predict suitable crop based on user input
6. *(Optional)* **Deployment using Streamlit**

---

## ✅ Results

- The model performs well on unseen data with high accuracy.
- Random Forest outperformed other models like SVM, Decision Tree, and KNN.
- User can input real-time values and get instant crop recommendations.
