# Bank Marketing Prediction

## Project Overview
This project is a Machine Learning web application that predicts whether a customer will subscribe to a bank term deposit based on demographic, financial, and previous marketing campaign information.
The application is developed using Python, Flask, Scikit-learn, HTML, and CSS, and is deployed on PythonAnywhere.

---

## Features
- Predicts whether a customer will subscribe to a term deposit.
- Displays the confidence score of each prediction.
- Responsive and user-friendly web interface.
- Random Forest Classifier for prediction.
- Flask-based web application.
- Deployed online using PythonAnywhere.
---

## Technologies Used

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- HTML
- CSS
- Git
- GitHub
- PythonAnywhere

---

## Project Structure

```text
Bank_Marketing_Pred/
│
├── app.py
├── bank_marketing_data.pkl
├── bank_additional_data.csv
├── requirements.txt
├── README.md
│
├── templates/
│   └── index.html
│
└── static/
    └── style.css
```
---

## Machine Learning Workflow

1. Data Collection
2. Data Preprocessing
3. Feature Engineering
4. Ordinal Encoding
5. One-Hot Encoding
6. Model Training
7. Model Evaluation
8. Model Serialization
9. Flask Integration
10. Deployment

---

## Model Information
**Algorithm**

- Random Forest Classifier

**Encoders**

- Ordinal Encoder
- One-Hot Encoder
- Label Encoder

**Model File**

- text
  bank_marketing_data.pkl

---

## Installation
Clone the repository:
bash
git clone https://github.com/zalimhash1/Bank_Marketing_Pred.git


Navigate to the project folder:
bash
cd Bank_Marketing_Pred


Install the required packages:
bash
pip install -r requirements.txt


Run the application:
bash
python app.py

Open your browser and visit:
text
http://127.0.0.1:5000
---

## Live Demo
https://zalim09.pythonanywhere.com

---

## Input Features
The model uses the following customer information:

- Age
- Job
- Marital Status
- Education
- Default
- Housing Loan
- Personal Loan
- Contact Type
- Month
- Day of Week
- Duration
- Campaign
- Previous Contacts
- Previous Outcome
- Employment Variation Rate
- Consumer Price Index
- Consumer Confidence Index
- Euribor 3 Month Rate
- Number of Employees

---

## Output
The application predicts one of the following:

- Customer Will Subscribe
- Customer Will NOT Subscribe
It also displays the prediction confidence score.

---

## Dataset
This project uses the Bank Marketing Dataset.
Source:
https://archive.ics.uci.edu/ml/datasets/bank+marketing

---

## Future Improvements
- Improve the user interface.
- Add feature importance visualization.
- Compare multiple Machine Learning models.
- Store prediction history.
- Generate downloadable prediction reports.

---

## Author
Muhammad-Din
GitHub:
https://github.com/zalimhash1

---

## License
This project is created for educational and portfolio purposes.
