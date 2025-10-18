new_trip = pd.DataFrame({
    'distance_km': [5.2],
    'passenger_count': [2],
    'hour': [18],
    'day_of_week': [4],  # Friday
    'month': [10]
})

# Scale features
new_trip_scaled = scaler.transform(new_trip)

# Predict fare
predicted_fare = model.predict(new_trip_scaled)
print(f"Predicted Fare: ${predicted_fare[0]:.2f}")
