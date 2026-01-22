# Titanic Survival Prediction System

## Project Overview

This project is a Machine Learning-based web application that predicts whether a passenger survived the Titanic disaster. It uses the **Titanic Dataset** and a **Random Forest Classifier** model trained on key passenger features. The system is served via a **Flask** web interface with a modern, glassmorphism design.

## Features

- **Machine Learning Model**: Random Forest Classifier.
- **Input Features**:
  - Passenger Class (1st, 2nd, 3rd)
  - Gender
  - Age
  - Number of Siblings/Spouses (SibSp)
  - Fare
- **Web GUI**: Responsive premium interface for easy interaction.
- **Backend**: Python (Flask).

## Project Structure

```
Titanic_Project_Adelodun_Abraham_23CG034020/
│
├── app.py                      # Flask Application
├── requirements.txt            # Python Dependencies
├── Titanic_hosted_webGUI_link.txt # Submission Details
├── README.md                   # Project Documentation
│
├── model/                      # ML Model Directory
│   ├── model_building.py       # Training Script
│   └── titanic_survival_model.pkl # Trained Model
│
├── static/                     # Assets
│   ├── style.css               # Styling
│   └── fluent_bg.svg           # Background
│
└── templates/                  # HTML Templates
    └── index.html              # Frontend
```

## Setup & Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Train Model (Optional)**:
   ```bash
   python model/model_building.py
   ```
3. **Run Application**:
   ```bash
   python app.py
   ```
4. **Access**: `http://127.0.0.1:5000/`

## Details

- **Student**: Adelodun Abraham (23CG034020)
- **Algorithm**: Random Forest
