import pandas as pd
import matplotlib
matplotlib.use('Agg')  # or another backend like 'Qt5Agg'
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Load the data/file with stats
file_path = r"C:/Users/ejesp/OneDrive/Desktop/PhilliesProj - Copy/obp.csv" 
data = pd.read_csv(file_path)

# Calulate the weighted OBP (Refrenced(Not exactly): https://triplesalley.wordpress.com/2010/12/22/marcel-and-forecasting-systems/)(Marcel Forcasting)
def calculate_weighted_obp(row):
    total_weighted_obp = 0  
    total_pa = 0  
    
    
    for year in range(16, 21): #Season '16 to Season '20
        pa_col = f'PA_{year}'
        obp_col = f'OBP_{year}'
        
        # Doesn't account years with no data
        if pd.notna(row[pa_col]) and pd.notna(row[obp_col]):
            total_weighted_obp += row[obp_col] * row[pa_col]
            total_pa += row[pa_col]
    
    # If there's no PA, return NaN
    if total_pa == 0:
        return np.nan
    
    # Return weighted OBP
    return total_weighted_obp / total_pa

# Apply the weighted OBP
data['Weighted_OBP'] = data.apply(calculate_weighted_obp, axis=1)

# Account for PA
data['Total_PA'] = data[['PA_16', 'PA_17', 'PA_18', 'PA_19', 'PA_20']].sum(axis=1)

# Filter out rows where the weighted OBP or the target OBP_21 is missing
data = data[pd.notna(data['Weighted_OBP']) & pd.notna(data['OBP_21'])]

# Separate Weighted OBP, Total PA, and OBP_21
X = data[['Weighted_OBP', 'Total_PA']]  
y = data['OBP_21']

# Machine learning: Split training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a RandomForestRegressor(Used instead of Linear Model because outliers were being projected wrongly(Trout, Soto, etc.))
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict OBP_21 
y_pred = model.predict(X_test)

# Compares predicted and observed values
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
data['mse'] = mse


### Created Plots and Tables ###
#Link with how I started developing models(https://stackoverflow.com/questions/49992300/python-how-to-show-graph-in-visual-studio-code-itself)

# 1. Residual Plot: Shows prediction error
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residual Plot (Prediction Error)')
plt.xlabel('Predicted OBP_21')
plt.ylabel('Residuals (Actual - Predicted)')
plt.savefig('residual_plot.png')
plt.show()

# 2. Predicted vs Actual Distribution Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.title('Actual vs Predicted OBP for 2021')
plt.xlabel('Actual OBP_21')
plt.ylabel('Predicted OBP_21')
plt.savefig('predicted_vs_actual.png')
plt.show()

### 4. Create and Save a New Table ###

data['Predicted_OBP_21'] = model.predict(X)

columns_to_save = ['Name', 'PA_16', 'OBP_16', 'PA_17', 'OBP_17', 'PA_18', 'OBP_18', 
                   'PA_19', 'OBP_19', 'PA_20', 'OBP_20', 'OBP_21','mse', 'Weighted_OBP', 'Predicted_OBP_21']

# Save this new table to a CSV file
output_file_path = "player_obp_comparison.csv"
data[columns_to_save].to_csv(output_file_path, index=False)
print(f"Player comparison table saved to {output_file_path}")
