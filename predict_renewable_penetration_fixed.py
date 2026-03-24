
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Synthetic dataset generation (more realistic with trends)
np.random.seed(42)
n = 1000
timestamps = pd.date_range(start="2023-01-01", periods=n, freq="H")

# Add diurnal cycle for solar irradiance
hours = np.arange(n) % 24
solar_cycle = np.clip(1000 * np.sin((hours/24) * np.pi), 0, None)
solar_irradiance = solar_cycle + np.random.normal(0, 50, n)

# Wind speed with some autocorrelation
wind_speed = np.cumsum(np.random.normal(0, 0.2, n)) + 7
wind_speed = np.clip(wind_speed, 0, None)

# Renewable gen depends on wind + solar irradiance
renewable_gen = 0.5 * wind_speed * 10 + 0.05 * solar_irradiance + np.random.normal(0, 20, n)
renewable_gen = np.clip(renewable_gen, 50, None)

total_gen = renewable_gen + np.random.uniform(200, 800, n)
demand = total_gen + np.random.uniform(-50, 50, n)

data = pd.DataFrame({
    "timestamp": timestamps,
    "renewable_gen": renewable_gen,
    "total_gen": total_gen,
    "demand": demand,
    "wind_speed": wind_speed,
    "solar_irradiance": solar_irradiance
})

# Calculate renewable penetration rate safely
eps = 1e-6
data["penetration_rate"] = data["renewable_gen"] / (data["total_gen"].clip(lower=eps))

# Sort by timestamp and create lag features (to prevent leakage)
data = data.sort_values("timestamp").reset_index(drop=True)
data["ren_lag1"] = data["renewable_gen"].shift(1)
data["tot_lag1"] = data["total_gen"].shift(1)
data["hour"] = data["timestamp"].dt.hour
data["dayofyear"] = data["timestamp"].dt.dayofyear

data = data.dropna().reset_index(drop=True)

# Feature set: only lagged + exogenous (no direct contemporaneous leakage)
X = data[["ren_lag1","tot_lag1","demand","wind_speed","solar_irradiance","hour","dayofyear"]]
y = data["penetration_rate"]

# Chronological split (not shuffled)
cut = int(0.8 * len(data))
X_train, X_test = X.iloc[:cut], X.iloc[cut:]
y_train, y_test = y.iloc[:cut], y.iloc[cut:]

# Models
lin_reg = LinearRegression()
rf_reg = RandomForestRegressor(n_estimators=200, random_state=42)

# Train models
lin_reg.fit(X_train, y_train)
rf_reg.fit(X_train, y_train)

# Predictions
y_pred_lin = lin_reg.predict(X_test)
y_pred_rf = rf_reg.predict(X_test)

# Evaluation
print("Linear Regression MAE:", mean_absolute_error(y_test, y_pred_lin))
print("Linear Regression MSE:", mean_squared_error(y_test, y_pred_lin))
print("Linear Regression R²:", r2_score(y_test, y_pred_lin))

print("Random Forest MAE:", mean_absolute_error(y_test, y_pred_rf))
print("Random Forest MSE:", mean_squared_error(y_test, y_pred_rf))
print("Random Forest R²:", r2_score(y_test, y_pred_rf))

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(y_test.values[:100], label="Actual", marker="o")
plt.plot(y_pred_lin[:100], label="Linear Predicted", linestyle="--")
plt.plot(y_pred_rf[:100], label="RF Predicted", linestyle="--")
plt.xlabel("Sample Index")
plt.ylabel("Penetration Rate")
plt.title("Renewable Penetration Rate Prediction (Fixed Pipeline)")
plt.legend()
plt.tight_layout()
plt.show()
