# Heart Disease Prediction Project

## Abstract

Heart disease remains one of the leading causes of mortality worldwide, driving the need for rapid and accurate early detection methods. This project demonstrates a machine learning approach using Logistic Regression to predict heart disease from clinical data. The system includes data loading, preprocessing, model training with an accuracy graph, and real-time prediction through a command-line interface. The design focuses on clear modularity and reproducibility, making it a solid baseline for future exploration.

## Introduction

Cardiovascular diseases, particularly heart disease, impose a significant health burden globally. Early detection and risk assessment are vital for reducing mortality through proactive interventions. The project leverages machine learning techniques to analyze patient data (e.g., age, cholesterol levels, blood pressure) and estimate heart disease risk.

Key highlights of the project:

- **Data Preprocessing:** Scaling, cleaning, and imputation for consistency.
- **Model Training:** Using Logistic Regression to build an interpretable binary classifier.
- **Evaluation:** Visual accuracy comparison through a generated graph.
- **Real-Time Prediction:** A user-friendly command-line interface for live predictions.

This project is designed as a modular and extensible solution for predictive analytics in healthcare.

## File Structure

The project is organized as follows:

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://your-repository-url.git
   cd Heart-Disease-Prediction
   ```

2. **Set Up the Virtual Environment (Windows)**

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

   For macOS/Linux:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   > If you don't have a `requirements.txt` file, ensure you install the following packages:
   >
   > - pandas
   > - scikit-learn
   > - matplotlib
   > - joblib

## Usage

### 1. Training the Model

Before running predictions, you must train the model on the dataset. Place your dataset (e.g., `Heart Disease.csv`) in the project folder (or update the path in `train.py`). Then execute:

````bash
python src/train.py

### 1. run the Model
```bash
puthon main.py
````
