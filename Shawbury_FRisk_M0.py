import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

# Load the dataset from Excel file with error handling
try:
    data = pd.read_excel('Shawbury_NewDataset.xlsx')
except FileNotFoundError:
    print("Error: The specified file was not found.")
    exit(1)
except Exception as e:
    print(f"An error occurred: {e}")
    exit(1)

# Preprocess the data
try:
    data['Month'] = pd.Categorical(data['Month']).codes
    X = data[['Month', 'Averagerainfall']]
    y = data['Groundwater']
except KeyError as e:
    print(f"Error: Missing column in the dataset - {e}")
    exit(1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (for SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter tuning for Random Forest and SVM
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

svm_param_grid = {
    'C': [0.1, 1, 10],
    'epsilon': [0.1, 0.5, 1.0]
}

# Define regression models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': GridSearchCV(RandomForestRegressor(random_state=42), rf_param_grid, cv=5),
    'SVM': GridSearchCV(SVR(kernel='linear'), svm_param_grid, cv=5)
}

# Train and evaluate models
results = []
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    results.append([model_name, mse, r2, mae])

    # Visualize Actual vs Predicted
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.title(f'Actual vs Predicted for {model_name}')
    plt.xlabel('Actual Groundwater Levels')
    plt.ylabel('Predicted Groundwater Levels')
    plt.grid()
    plt.show()

    # Visualize Residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 5))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f'Residuals for {model_name}')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.grid()
    plt.show()

    # Print evaluation metrics for each model
    print(f"Model: {model_name}")
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print(f"  R-squared: {r2:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae:.4f}")

# Creating DataFrame to display results
results_df = pd.DataFrame(results, columns=['Model', 'MSE', 'R-squared', 'MAE'])

# Visualizing model performance
plt.figure(figsize=(10, 6))
results_df.plot(x='Model', y=['MSE', 'R-squared', 'MAE'], kind='bar', rot=0)
plt.title('Model Performance Comparison')
plt.ylabel('Metric Score')
plt.show()

# Packaging best performing model with joblib
best_model_name = results_df.loc[results_df['R-squared'].idxmax()]['Model']
best_model = models[best_model_name]
joblib.dump(best_model, 'groundwater_prediction_model.joblib')
print(f'Best model: {best_model_name}')
print('Model saved to groundwater_prediction_model.joblib')

# Load and use the saved model
loaded_model = joblib.load('groundwater_prediction_model.joblib')

# Example: Predict groundwater level for new data
new_data = pd.DataFrame([[10, 150]], columns=['Month', 'Averagerainfall'])
new_data = scaler.transform(new_data)  # Apply the same scaling as used during training
groundwater_level = loaded_model.predict(new_data)
print(f'Predicted Groundwater Level: {groundwater_level[0]:.4f}')