import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset from an Excel file
try:
    data = pd.read_excel('Shawbury_NewDataset.xlsx')  # Replace with your actual file name
except FileNotFoundError:
    print("Error: The specified file was not found.")
    exit(1)
except Exception as e:
    print(f"An error occurred: {e}")
    exit(1)

# Check the first few rows of the dataset
print(data.head())

# Set the plot style
sns.set(style="whitegrid")

# 1. Bar Chart: Average Groundwater Level by Month
avg_groundwater_by_month = data.groupby('Month')['Groundwater'].mean().reset_index()
plt.figure(figsize=(10, 5))
sns.barplot(x='Month', y='Groundwater', data=avg_groundwater_by_month)
plt.title('Average Groundwater Level by Month')
plt.ylabel('Average Groundwater Level')
plt.xlabel('Month')
plt.xticks(rotation=45)
plt.show()

# 2. Line Chart: Groundwater Level Over Time
plt.figure(figsize=(10, 5))
sns.lineplot(x='Month', y='Groundwater', data=data, marker='o')
plt.title('Groundwater Level Over Months')
plt.ylabel('Groundwater Level')
plt.xlabel('Month')
plt.xticks(rotation=45)
plt.show()

# 3. Column Chart: Total Average Rainfall by Month
total_rainfall_by_month = data.groupby('Month')['Averagerainfall'].sum().reset_index()
plt.figure(figsize=(10, 5))
sns.barplot(x='Month', y='Averagerainfall', data=total_rainfall_by_month)
plt.title('Total Average Rainfall by Month')
plt.ylabel('Total Average Rainfall')
plt.xlabel('Month')
plt.xticks(rotation=45)
plt.show()

# 4. Scatter Plot: Relationship Between Average Rainfall and Groundwater Level
plt.figure(figsize=(10, 5))
sns.scatterplot(x='Averagerainfall', y='Groundwater', data=data)
plt.title('Relationship Between Average Rainfall and Groundwater Level')
plt.xlabel('Average Rainfall')
plt.ylabel('Groundwater Level')
plt.show()

# 5. Pair Plot: To visualize all combinations of features
sns.pairplot(data)
plt.title('Pair Plot of the Features')
plt.show()