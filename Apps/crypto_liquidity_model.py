# INSTALL NECESSARY PACKAGES

!pip install pandas numpy matplotlib seaborn scikit-learn --quiet

# IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import pickle
import os # Import os to check file size

import warnings
warnings.filterwarnings("ignore")

from google.colab import files
from IPython.display import display # Import display for showing dataframes

uploaded = files.upload()

# Read uploaded files into dataframes

df1 = None
df2 = None

# A simple way to handle potentially multiple files uploaded:
# Read all uploaded CSVs into a list of dataframes

uploaded_dataframes = []
for filename in uploaded.keys():
    try:
        # Assuming the uploaded files are CSVs
        df_temp = pd.read_csv(filename)
        uploaded_dataframes.append(df_temp)
    except Exception as e:
        print(f"Error reading file {filename}: {e}")

# Combine data
# Check if enough dataframes were loaded to proceed with concatenation

if len(uploaded_dataframes) >= 2:

    df1 = uploaded_dataframes[0]
    df2 = uploaded_dataframes[1]
    df = pd.concat([df1, df2], ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by=['coin', 'date'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print("Data combined successfully.")
    display(df.head()) # Use display to show the head of the combined dataframe
else:
    print("Error: Not enough dataframes loaded from uploaded files to concatenate.")
    # Exit or handle the error appropriately if concatenation is essential
    # For now, let's stop execution if data loading fails
    exit()


# DATA PREPROCESSING

df.dropna(inplace=True)
for col in ['1h', '24h', '7d']:
    # Ensure columns exist before converting type
    if col in df.columns:
        df[col] = df[col].astype(float)
    else:
        print(f"Warning: Column '{col}' not found in DataFrame.")


# Check types
print("\nDataFrame dtypes after preprocessing:")
print(df.dtypes)

# FEATURE ENGINEERING

# Ensure columns exist before creating new features
if '24h_volume' in df.columns and 'mkt_cap' in df.columns:
    df['liquidity_ratio'] = df['24h_volume'] / df['mkt_cap']
else:
    print("Warning: '24h_volume' or 'mkt_cap' column not found for liquidity_ratio.")
    # Handle the case where columns are missing
    if 'liquidity_ratio' not in df.columns:
         df['liquidity_ratio'] = np.nan # Create column with NaNs to avoid downstream errors


if '24h' in df.columns:
    df['volatility'] = df['24h'].abs()
else:
     print("Warning: '24h' column not found for volatility.")
     if 'volatility' not in df.columns:
         df['volatility'] = np.nan # Create column with NaNs


# Ensure the columns used for head() exist
display_cols = [col for col in ['coin', 'liquidity_ratio', 'volatility'] if col in df.columns]
if display_cols:
    print("\nHead of feature engineered columns:")
    display(df[display_cols].head())
else:
    print("Warning: None of the display columns ('coin', 'liquidity_ratio', 'volatility') found.")


# EDA - Heatmap of correlations
# Filter features to only include those present in the dataframe
heatmap_features = [col for col in ['price', '24h_volume', 'mkt_cap', 'liquidity_ratio', 'volatility'] if col in df.columns]
if len(heatmap_features) > 1: # Need at least two columns for correlation
    plt.figure(figsize=(10, 6))
    sns.heatmap(df[heatmap_features].corr(), annot=True, cmap='coolwarm')
    plt.title("Feature Correlation Heatmap")
    plt.show()
else:
    print("Warning: Not enough features available to plot correlation heatmap.")


# MODEL DEVELOPMENT - Random Forest Regressor
# Update features list based on available columns after preprocessing and feature engineering
# Ensure target column exists
features = [col for col in ['price', '1h', '24h', '7d', '24h_volume', 'mkt_cap', 'volatility'] if col in df.columns]
target = 'liquidity_ratio'

if target not in df.columns:
    print(f"Error: Target column '{target}' not found in DataFrame.")
    exit() # Or raise

# Keep only features that are in the dataframe and are numeric (required by model)
X = df[features].select_dtypes(include=np.number)
y = df[target]

# Filter features again based on available numeric columns in X
features = X.columns.tolist()
if not features:
    print("Error: No numeric features available for model training.")
    exit() # Or raise

print(f"\nUsing features for model: {features}")

# Drop rows with NaNs that might have been introduced during feature engineering if columns were missing
# Or handle NaNs in features/target appropriately for the model
X.dropna(inplace=True)
y = y[X.index] # Align y with cleaned X

if X.empty:
    print("Error: No data remaining after dropping NaNs for model training.")
    exit() # Or raise

print(f"\nShape of X after dropping NaNs: {X.shape}")
print(f"Shape of y after dropping NaNs: {y.shape}")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nShape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")


model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.model_selection import GridSearchCV

params = {'n_estimators': [100, 200], 'max_depth': [5, 10, None]}
grid = GridSearchCV(RandomForestRegressor(), params, cv=3, scoring='r2')
grid.fit(X_train, y_train)

print("Best Params:", grid.best_params_)


# MODEL EVALUATION
# Ensure there are predictions to evaluate
if len(y_test) > 0:
    print("\n R² Score:", r2_score(y_test, y_pred))
    # Calculate MSE and then take the square root for RMSE
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(" RMSE:", rmse)
else:
    print("Warning: No test data available for model evaluation.")


# SAVE MODEL FOR DEPLOYMENT
try:
    model_filename = "crypto_liquidity_model.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved successfully to {model_filename}.")
    # Optional: Check file size
    if os.path.exists(model_filename):
        print(f"Model file size: {os.path.getsize(model_filename)} bytes")
except Exception as e:
    print(f"\nError saving model: {e}")
