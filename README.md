# 🎬 IMDB Movie Review Sentiment Analysis (RNN + Streamlit)

This is a complete **Deep Learning NLP project** that classifies IMDB movie reviews as **Positive 😊** or **Negative 😞** using a **Simple RNN model**.  
It is deployed with an **interactive Streamlit dashboard** that includes **explainability**, **confidence visualization**, and **input diagnostics**.

---

## 🚀 Live Features

- 🧠 **Deep Learning Model**: Trained a Simple RNN on the IMDB dataset  
- ✍️ **Real-time Prediction**: Enter custom movie reviews  
- 📊 **Confidence Visualization**: Probability bar and progress indicator  
- 🔍 **Explainable AI**: Token-level contribution using Leave-One-Out analysis  
- 📝 **Text Diagnostics**: Token count and tokenized input inspection  
- ⚙️ **Configurable UI**: Adjustable sequence length and decision threshold  
- 📥 **Downloadable JSON Report**: For reproducibility and analysis  
- 🎯 **Recruiter-Friendly UI**: Clean layout, sidebar demos, dashboard-style output  

---

## 🖼️ Application Preview

**Dashboard Sections**
- ✍️ Review Input  
- 🎭 Sentiment Result  
- 📊 Model Confidence (Graph)  
- 🔎 Text Diagnostics (Side-by-Side)  
- 📋 Token Contribution Table  
- 📥 JSON Report Download  

---

## 🧠 Model Overview

- 📦 **Dataset**: IMDB Movie Reviews (Keras built-in)  
- 🎯 **Task**: Binary Sentiment Classification  
- 🏗️ **Architecture**:
  - Embedding Layer  
  - Simple RNN  
  - Dense Output Layer (Sigmoid)  
- 📉 **Loss Function**: Binary Crossentropy  
- ⚡ **Optimizer**: Adam  
- ✅ **Evaluation Metric**: Accuracy  

---

## 🔍 Explainability (Why this prediction?)

This project answers the question:

> **“Why did the model predict this sentiment?”**

### 🧩 Token Contribution (Leave-One-Out)

Each word is removed one at a time to measure how it affects the final prediction score.

- ▲ **Positive influence**  
- ▼ **Negative influence**  

This improves **model transparency, interpretability, and trust**.

---

## 📝 Input Diagnostics

- 🔢 Displays total token count  
- 🔠 Shows the first 60 tokens after preprocessing  
- 👁️ Helps users understand how raw text is interpreted by the model  

---

## ⚙️ Configuration Options (Sidebar)

- 📏 **Max Sequence Length**  
  Controls padding and truncation for RNN input  

- 🎚️ **Decision Threshold**  
  Adjusts model sensitivity for Positive vs Negative classification  

- ⚡ **Quick Demo Buttons**  
  One-click example reviews for instant demos  

---

## 📥 JSON Report (Reproducibility)

Each prediction can be downloaded as a structured **JSON report** containing:

```json
{
  "input": "User review text",
  "prediction": 0.82,
  "sentiment": "Positive",
  "top_token_importances": [...],
  "maxlen_used": 500,
  "threshold": 0.5
}
