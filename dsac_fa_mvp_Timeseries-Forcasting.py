from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import zscore
from google.cloud import firestore

def run_prophet_forecast(data, seasonality_mode, seasonality_prior_scale, changepoint_prior_scale):
    # Create a Prophet model with adjusted seasonality parameters
    model = Prophet(seasonality_mode=seasonality_mode,
                    seasonality_prior_scale=seasonality_prior_scale,
                    changepoint_prior_scale=changepoint_prior_scale,
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False)
    
    # Fit the model with the data
    model.fit(data)
    
    # Generate future dates for forecasting
    future = model.make_future_dataframe(periods=len(data))
    
    # Make predictions for the future dates
    forecast = model.predict(future)
    
    return forecast

def preprocess_data(data, date_column, target_column, normalize=False, standardize=False, handle_outliers=False):
    # Convert the date column to datetime format
    data[date_column] = pd.to_datetime(data[date_column])
    
    # Remove missing values
    data.dropna(inplace=True)
    
    # Rename the date column and target column
    data = data.rename(columns={date_column: 'ds', target_column: 'y'})
    
    # Handle outliers
    if handle_outliers:
        z_scores = zscore(data['y'])
        threshold = 3
        outliers = np.abs(z_scores) > threshold
        data = data[~outliers]
    
    # Normalize the target column
    if normalize:
        scaler = MinMaxScaler()
        data['y'] = scaler.fit_transform(data['y'].values.reshape(-1, 1))
    
    # Standardize the target column
    if standardize:
        scaler = StandardScaler()
        data['y'] = scaler.fit_transform(data['y'].values.reshape(-1, 1))
    
    return data

def split_train_test(data, train_ratio):
    # Split the dataset into train and test sets
    train = data[:int(train_ratio * len(data))]
    test = data[int(train_ratio * len(data)):]
    
    return train, test

def perform_hyperparameter_tuning(train_data, test_data, hyperparameter_grid):
    best_accuracy = np.inf
    best_params = {}

    # Perform hyperparameter tuning using grid search
    for params in ParameterGrid(hyperparameter_grid):
        # Perform time series forecasting with current hyperparameters
        forecast = run_prophet_forecast(train_data, **params)

        # Extract the predicted values and actual values for comparison
        predicted = forecast['yhat']
        actual = test_data['y']

        # Calculate the model accuracy score (Mean Absolute Error)
        accuracy = mean_absolute_error(actual[:len(test_data)], predicted[:len(test_data)])

        # Update the best parameters if the accuracy improves
        if accuracy < best_accuracy:
            best_accuracy = accuracy
            best_params = params
    
    return best_params

def perform_forecast(data, date_column, target_column, train_ratio, hyperparameter_grid, normalize=False, standardize=False, handle_outliers=False):
    # Preprocess the data
    data = preprocess_data(data, date_column, target_column, normalize, standardize, handle_outliers)
    
    # Split the data into train and test sets
    train_data, test_data = split_train_test(data, train_ratio)
    
    # Perform hyperparameter tuning
    best_params = perform_hyperparameter_tuning(train_data, test_data, hyperparameter_grid)
    
    # Perform final time series forecasting with the best hyperparameters
    forecast = run_prophet_forecast(train_data, **best_params)
    
    # Extract the predicted values and actual values for comparison
    predicted = forecast['yhat']
    actual = test_data['y']
    
    # Calculate the model accuracy score (Mean Absolute Error)
    accuracy = mean_absolute_error(actual[:len(test_data)], predicted[:len(test_data)])
    
    # Calculate the percentage accuracy
    percentage_accuracy = (1 - (accuracy / np.mean(actual[:len(test_data)]))) * 100
    
    # Plot the predicted values and actual values for comparison
    plot_forecast(test_data['ds'], predicted[:len(test_data)], actual[:len(test_data)], target_column)

    # Print the model accuracy score and best hyperparameters
    print("Best Hyperparameters:", best_params)
    print(f"{target_column} Accuracy Score (Mean Absolute Error):", accuracy)
    print(f"{target_column} Percentage Accuracy:", percentage_accuracy)

def define_hyperparameter_grid():
    hyperparameter_grid = {
        'seasonality_mode': ['multiplicative', 'additive'],
        'seasonality_prior_scale': [10, 20, 30],
        'changepoint_prior_scale': [0.01, 0.05, 0.1]
    }
    return hyperparameter_grid

def retrieve_data_from_firestore(collection_name):
    # Initialize Firestore client
    db = firestore.Client()

    # Retrieve data from Firestore collection
    data = []
    collection_ref = db.collection(collection_name)
    documents = collection_ref.stream()
    for doc in documents:
        data.append(doc.to_dict())
    
    # Convert data to DataFrame
    data_df = pd.DataFrame(data)
    
    return data_df

def plot_forecast(date, predicted, actual, target_column):
    plt.figure(figsize=(10, 6))
    plt.plot(date, predicted, label='Predicted')
    plt.plot(date, actual, label='Actual')
    plt.title(f'{target_column} - Predicted vs Actual')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

# Usage example:
collection_name = 'your_collection_name'
data = retrieve_data_from_firestore(collection_name)

# file_path = 'ford_data.csv'
# data = load_data_from_csv(file_path)

date_column = 'Date'
target_column = 'Close'
train_ratio = 0.8
hyperparameter_grid = define_hyperparameter_grid()

# Perform forecast without normalization or standardization
perform_forecast(data, date_column, target_column, train_ratio, hyperparameter_grid)

# Perform forecast with normalization
perform_forecast(data, date_column, target_column, train_ratio, hyperparameter_grid, normalize=True)

# Perform forecast with standardization
perform_forecast(data, date_column, target_column, train_ratio, hyperparameter_grid, standardize=True)

# Perform forecast with outlier handling
perform_forecast(data, date_column, target_column, train_ratio, hyperparameter_grid, handle_outliers=True)
