\# 💳 CreditGuard-AI – Credit Risk Prediction API



CreditGuard-AI is a production-ready machine learning API that predicts the credit risk of loan applicants using a trained classification model. The system is fully containerized using Docker and designed for real-world deployment scenarios.



\---



\## 🚀 Features



\- 📊 Predicts credit risk (High / Low) for loan applicants

\- 🤖 Machine Learning model for classification

\- 🧹 End-to-end preprocessing pipeline

\- ⚡ FastAPI-based REST API

\- 🐳 Fully Dockerized for portability and deployment

\- 📦 Clean modular code structure



\---



\## 🧠 Problem Statement



Financial institutions need to assess whether a loan applicant is likely to default. This project builds a machine learning model that evaluates applicant data and predicts credit risk, enabling better decision-making.



\---



\## 🛠️ Tech Stack



\- Python

\- FastAPI

\- Scikit-learn

\- Pandas

\- NumPy

\- Docker



\---



\## 🏗️ System Architecture



Input Data → Preprocessing → Feature Engineering → Model → Prediction API → Response


\## Dataset Installation: Datasets can be installed from https://www.kaggle.com/competitions/home-credit-default-risk/data

\### Training dataset: application_train.csv 
\### Testing dataset: application_test.csv


\---



\## 📦 Installation



\### 1. Clone the repository



```bash

git clone https://github.com/your-username/CreditGuard-AI.git

cd CreditGuard-AI



\### 2. Running without Docker



\#### a. pip install -r requirements.txt



\#### b. python app.py



\### 3. Install Docker



\### 4. Build Docker Image



```bash

docker build -t credit-risk-api .



\### 5. Run the Container



docker run -p 5000:5000 credit-risk-api



\### 6. Test the api



\#### a. open Swagger UI / Postman



\#### b. Test endpoint: **http://127.0.0.1:5000/predict**



Example input data: \[{

&#x20; "SK\_ID\_CURR": 100001,

&#x20; "NAME\_CONTRACT\_TYPE": "Cash loans",

&#x20; "CODE\_GENDER": "M",

&#x20; "FLAG\_OWN\_CAR": "Y",

&#x20; "FLAG\_OWN\_REALTY": "Y",

&#x20; "CNT\_CHILDREN": 1,

&#x20; "AMT\_INCOME\_TOTAL": 180000,

&#x20; "AMT\_CREDIT": 500000,

&#x20; "AMT\_ANNUITY": 25000,

&#x20; "AMT\_GOODS\_PRICE": 450000,

&#x20; "NAME\_TYPE\_SUITE": "Family",

&#x20; "NAME\_INCOME\_TYPE": "Working",

&#x20; "NAME\_EDUCATION\_TYPE": "Higher education",

&#x20; "NAME\_FAMILY\_STATUS": "Married",

&#x20; "NAME\_HOUSING\_TYPE": "House / apartment",

&#x20; "REGION\_POPULATION\_RELATIVE": 0.02,

&#x20; "DAYS\_BIRTH": -12000,

&#x20; "DAYS\_EMPLOYED": -4000,

&#x20; "DAYS\_REGISTRATION": -5000,

&#x20; "DAYS\_ID\_PUBLISH": -3000,

&#x20; "OWN\_CAR\_AGE": 5,

&#x20; "FLAG\_MOBIL": 1,

&#x20; "FLAG\_EMP\_PHONE": 1,

&#x20; "FLAG\_WORK\_PHONE": 1,

&#x20; "FLAG\_CONT\_MOBILE": 1,

&#x20; "FLAG\_PHONE": 1,

&#x20; "FLAG\_EMAIL": 1,

&#x20; "OCCUPATION\_TYPE": "Laborers",

&#x20; "CNT\_FAM\_MEMBERS": 3,

&#x20; "REGION\_RATING\_CLIENT": 2,

&#x20; "REGION\_RATING\_CLIENT\_W\_CITY": 2,

&#x20; "WEEKDAY\_APPR\_PROCESS\_START": "MONDAY",

&#x20; "HOUR\_APPR\_PROCESS\_START": 10,

&#x20; "REG\_REGION\_NOT\_LIVE\_REGION": 0,

&#x20; "REG\_REGION\_NOT\_WORK\_REGION": 0,

&#x20; "LIVE\_REGION\_NOT\_WORK\_REGION": 0,

&#x20; "REG\_CITY\_NOT\_LIVE\_CITY": 0,

&#x20; "REG\_CITY\_NOT\_WORK\_CITY": 0,

&#x20; "LIVE\_CITY\_NOT\_WORK\_CITY": 0,

&#x20; "ORGANIZATION\_TYPE": "Business Entity Type 3",

&#x20; "EXT\_SOURCE\_1": 0.5,

&#x20; "EXT\_SOURCE\_2": 0.6,

&#x20; "EXT\_SOURCE\_3": 0.55,

&#x20; "APARTMENTS\_AVG": 0.08,

&#x20; "BASEMENTAREA\_AVG": 0.05,

&#x20; "YEARS\_BEGINEXPLUATATION\_AVG": 0.98,

&#x20; "YEARS\_BUILD\_AVG": 0.75,

&#x20; "COMMONAREA\_AVG": 0.02,

&#x20; "ELEVATORS\_AVG": 0.1,

&#x20; "ENTRANCES\_AVG": 0.12,

&#x20; "FLOORSMAX\_AVG": 0.3,

&#x20; "FLOORSMIN\_AVG": 0.1,

&#x20; "LANDAREA\_AVG": 0.04,

&#x20; "LIVINGAPARTMENTS\_AVG": 0.07,

&#x20; "LIVINGAREA\_AVG": 0.1,

&#x20; "NONLIVINGAPARTMENTS\_AVG": 0.01,

&#x20; "NONLIVINGAREA\_AVG": 0.02,

&#x20; "APARTMENTS\_MODE": 0.08,

&#x20; "BASEMENTAREA\_MODE": 0.05,

&#x20; "YEARS\_BEGINEXPLUATATION\_MODE": 0.98,

&#x20; "YEARS\_BUILD\_MODE": 0.75,

&#x20; "COMMONAREA\_MODE": 0.02,

&#x20; "ELEVATORS\_MODE": 0.1,

&#x20; "ENTRANCES\_MODE": 0.12,

&#x20; "FLOORSMAX\_MODE": 0.3,

&#x20; "FLOORSMIN\_MODE": 0.1,

&#x20; "LANDAREA\_MODE": 0.04,

&#x20; "LIVINGAPARTMENTS\_MODE": 0.07,

&#x20; "LIVINGAREA\_MODE": 0.1,

&#x20; "NONLIVINGAPARTMENTS\_MODE": 0.01,

&#x20; "NONLIVINGAREA\_MODE": 0.02,

&#x20; "APARTMENTS\_MEDI": 0.08,

&#x20; "BASEMENTAREA\_MEDI": 0.05,

&#x20; "YEARS\_BEGINEXPLUATATION\_MEDI": 0.98,

&#x20; "YEARS\_BUILD\_MEDI": 0.75,

&#x20; "COMMONAREA\_MEDI": 0.02,

&#x20; "ELEVATORS\_MEDI": 0.1,

&#x20; "ENTRANCES\_MEDI": 0.12,

&#x20; "FLOORSMAX\_MEDI": 0.3,

&#x20; "FLOORSMIN\_MEDI": 0.1,

&#x20; "LANDAREA\_MEDI": 0.04,

&#x20; "LIVINGAPARTMENTS\_MEDI": 0.07,

&#x20; "LIVINGAREA\_MEDI": 0.1,

&#x20; "NONLIVINGAPARTMENTS\_MEDI": 0.01,

&#x20; "NONLIVINGAREA\_MEDI": 0.02,

&#x20; "FONDKAPREMONT\_MODE": "reg oper account",

&#x20; "HOUSETYPE\_MODE": "block of flats",

&#x20; "TOTALAREA\_MODE": 0.12,

&#x20; "WALLSMATERIAL\_MODE": "Panel",

&#x20; "EMERGENCYSTATE\_MODE": "No",

&#x20; "OBS\_30\_CNT\_SOCIAL\_CIRCLE": 1,

&#x20; "DEF\_30\_CNT\_SOCIAL\_CIRCLE": 0,

&#x20; "OBS\_60\_CNT\_SOCIAL\_CIRCLE": 1,

&#x20; "DEF\_60\_CNT\_SOCIAL\_CIRCLE": 0,

&#x20; "DAYS\_LAST\_PHONE\_CHANGE": -1000,

&#x20; "FLAG\_DOCUMENT\_2": 0,

&#x20; "FLAG\_DOCUMENT\_3": 1,

&#x20; "FLAG\_DOCUMENT\_4": 0,

&#x20; "FLAG\_DOCUMENT\_5": 0,

&#x20; "FLAG\_DOCUMENT\_6": 0,

&#x20; "FLAG\_DOCUMENT\_7": 0,

&#x20; "FLAG\_DOCUMENT\_8": 0,

&#x20; "FLAG\_DOCUMENT\_9": 0,

&#x20; "FLAG\_DOCUMENT\_10": 0,

&#x20; "FLAG\_DOCUMENT\_11": 0,

&#x20; "FLAG\_DOCUMENT\_12": 0,

&#x20; "FLAG\_DOCUMENT\_13": 0,

&#x20; "FLAG\_DOCUMENT\_14": 0,

&#x20; "FLAG\_DOCUMENT\_15": 0,

&#x20; "FLAG\_DOCUMENT\_16": 0,

&#x20; "FLAG\_DOCUMENT\_17": 0,

&#x20; "FLAG\_DOCUMENT\_18": 0,

&#x20; "FLAG\_DOCUMENT\_19": 0,

&#x20; "FLAG\_DOCUMENT\_20": 0,

&#x20; "FLAG\_DOCUMENT\_21": 0,

&#x20; "AMT\_REQ\_CREDIT\_BUREAU\_HOUR": 0,

&#x20; "AMT\_REQ\_CREDIT\_BUREAU\_DAY": 0,

&#x20; "AMT\_REQ\_CREDIT\_BUREAU\_WEEK": 1,

&#x20; "AMT\_REQ\_CREDIT\_BUREAU\_MON": 2,

&#x20; "AMT\_REQ\_CREDIT\_BUREAU\_QRT": 1,

&#x20; "AMT\_REQ\_CREDIT\_BUREAU\_YEAR": 3

}]

Expected output : \[{"prediction" : \[0]"}]

