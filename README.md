# 🚖 NYC Taxi Fare Analysis

## Overview
Predict taxi fares in New York City using trip details such as pickup/dropoff locations, timestamps, and passenger count. Includes EDA, feature engineering, and regression modeling.

## Output and predicted fare
<img width="713" height="470" alt="image" src="https://github.com/user-attachments/assets/53ae5e32-ac55-4701-8161-b8d708e5dab4" />
<img width="695" height="470" alt="image" src="https://github.com/user-attachments/assets/699faac6-0cb5-4c6e-b4bc-42c10ab88653" />
<img width="663" height="435" alt="image" src="https://github.com/user-attachments/assets/523deb50-ec0e-45f1-ae39-9d1c2607d18b" />
Predicted Fare: $17.07

## Dataset
Sourced from [Kaggle NYC Taxi Fare Prediction](https://www.kaggle.com/datasets/dansbecker/new-york-city-taxi-fare-prediction)
- `train.csv`: Historical trip data with fare amounts
- `test.csv`: Test trips for prediction
- `sample_submission.csv`: Submission template

## Features
- Geospatial: Haversine distance between pickup and dropoff
- Temporal: Hour, day of week, month extraction
- Other: Passenger count
- Modeling: RandomForestRegressor or LinearRegression

## Installation
```bash
git clone https://github.com/YourGitHubUsername/NYC-Taxi-Fare-Analysis.git
cd NYC-Taxi-Fare-Analysis
pip install -r requirements.txt
