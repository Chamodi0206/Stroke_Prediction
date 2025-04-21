# 🧠 Stroke Prediction Web App

This project is a machine learning-based web application that predicts the likelihood of a person having a stroke based on their health-related input data. The app is built using **Flask** for the backend and **scikit-learn** for the machine learning model.

---

## 🚀 Features

- Predicts stroke risk using health indicators  
- Trained on a dataset with features like age, gender, glucose level, BMI, hypertension, etc.  
- Includes one-hot encoded dropdowns for smoking status  
- Interactive and styled UI with a clean interface  

---

## 🛠 Technologies Used

- Python  
- Flask  
- scikit-learn  
- SMOTE for data balancing  
- RobustScaler for normalization  
- HTML/CSS for frontend  
- Pickle for model serialization  

---

## 📊 Dataset Info

The model was trained on a medical dataset with the following features:

- **Gender**  
- **Age**  
- **Hypertension** (0 = No, 1 = Yes)  
- **Heart Disease** (0 = No, 1 = Yes)  
- **Ever Married**  
- **Work Type**  
- **Residence Type** (Urban/Rural)  
- **Avg Glucose Level**  
- **BMI**  
- **Smoking Status** (formerly smoked, never smoked, smokes, Unknown)  

---

## 🔧 Installation & Setup

### 1. Clone the Repository

git clone https://github.com/Chamodi0206/stroke_prediction_apll.git
cd stroke_prediction_app


2. Install Dependencies
It's recommended to create a virtual environment first.

bash
Copy code
pip install -r requirements.txt
If you don’t have a requirements.txt, install manually:

pip install flask scikit-learn numpy

3. Ensure Your Model is Saved
Make sure RFC_model.pkl exists in the root directory. You can replace it with another trained model if desired.

4. Run the Flask App
python app.py

6. Access in Browser
Go to: http://127.0.0.1:5000


📌 Notes
Ensure the input encoding in the web form matches what the model expects.

One-hot encoding is used for smoking status.




