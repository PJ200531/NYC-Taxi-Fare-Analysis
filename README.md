# 🚖 NYC Taxi Fare Analysis

## Overview
Predict taxi fares in New York City using trip details such as pickup/dropoff locations, timestamps, and passenger count. Includes EDA, feature engineering, and regression modeling.

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
