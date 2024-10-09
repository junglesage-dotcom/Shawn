import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # Import Linear Regression
from sklearn.ensemble import RandomForestRegressor  # Import Random Forest Regressor
from sklearn.svm import SVR  # Import Support Vector Regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

# Load the dataset from Excel file
data = pd.read_excel('Shawbury_NewDataset.xlsx')  # Replace 'flood_data.xlsx' with your actual file name

# Preprocess the data
# Convert categorical features to numerical
data['Month'] = pd.Categorical(data['Month']).codes

# Separate features and target variable
X = data[['Month', 'Averagerainfall']]
y = data['Groundwater']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (optional but recommended for SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define regression models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'SVM': SVR(kernel='linear')
}

# Train and evaluate models
results = []
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append([model_name, mse, r2])

# Create a dataframe to display the results
results_df = pd.DataFrame(results, columns=['Model', 'MSE', 'R-squared'])

# Visualize model performance
plt.figure(figsize=(10, 6))
results_df.plot(x='Model', y=['MSE', 'R-squared'], kind='bar', rot=0)
plt.title('Model Performance Comparison')
plt.ylabel('Metric Score')
plt.show()

# Package the best performing model with joblib
best_model_name = results_df.loc[results_df['R-squared'].idxmax()]['Model']
best_model = models[best_model_name]
joblib.dump(best_model, 'groundwater_prediction_model.joblib')
print(f'Best model: {best_model_name}')
print('Model saved to groundwater_prediction_model.joblib')

# Load and use the saved model
loaded_model = joblib.load('groundwater_prediction_model.joblib')

# Example: Predict groundwater level for new data
new_data = pd.DataFrame([[10, 150]], columns=['Month', 'Averagerainfall'])
new_data = scaler.transform(new_data)  # Apply same scaling as used during training
groundwater_level = loaded_model.predict(new_data)
print(f'Predicted Groundwater level: {groundwater_level[0]}')
