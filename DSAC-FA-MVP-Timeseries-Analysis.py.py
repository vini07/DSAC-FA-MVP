import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_percentage_error

class StockPredictionLSTM:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.df = None
        self.scaler = MinMaxScaler()
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.model = None
        self.mape_results = []
    
    def download_data(self):
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
    
    def preprocess_data(self, features):
        self.df = self.data[features]
        self.scaled_data = self.scaler.fit_transform(self.df)
    
    def split_data(self, train_size):
        train_size = int(len(self.scaled_data) * train_size)
        self.train_data = self.scaled_data[:train_size]
        self.test_data = self.scaled_data[train_size:]
    
    def create_sequences(self, data, sequence_length):
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i+sequence_length])
            y.append(data[i+sequence_length])
        X = np.array(X)
        y = np.array(y)
        return X, y
    
    def build_model(self, sequence_length):
        self.model = Sequential()
        self.model.add(LSTM(128, return_sequences=True, input_shape=(sequence_length, self.df.shape[1])))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(128, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(64, return_sequences=False))
        self.model.add(Dense(64))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(self.df.shape[1]))
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
    def train_model(self, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
        early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
        self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])
    
    def predict(self, X_test):
        predicted_data = self.model.predict(X_test)
        predicted_data = self.scaler.inverse_transform(predicted_data)
        return predicted_data
    
    def calculate_mape(self, true_data, predicted_data):
        mape = mean_absolute_percentage_error(true_data, predicted_data) * 100
        return mape
    
    def plot_results(self, true_data, predicted_data, sequence_length):
        plt.figure(figsize=(10, 6))
        plt.plot(true_data, label='True Price')
        plt.plot(predicted_data, label='Predicted Price')
        plt.title(f'True vs Predicted Price (Seq {sequence_length}) for {self.ticker}')
        plt.xlabel('Day')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
    
    def run(self, sequence_lengths, train_size=0.8, epochs=100, batch_size=32):
        self.download_data()
        
        for sequence_length in sequence_lengths:
            self.preprocess_data(['Open', 'Close'])
            self.split_data(train_size)
            
            self.X_train, self.y_train = self.create_sequences(self.train_data, sequence_length)
            self.X_test, self.y_test = self.create_sequences(self.test_data, sequence_length)
            
            self.build_model(sequence_length)
            self.train_model(self.X_train, self.y_train, self.X_test, self.y_test, epochs, batch_size)
            
            predicted_data = self.predict(self.X_test)
            true_data = self.scaler.inverse_transform(self.y_test)
            
            mape = self.calculate_mape(true_data, predicted_data)
            self.mape_results.append(mape)
            
            self.plot_results(true_data[:, 0], predicted_data[:, 0], sequence_length)
            self.plot_results(true_data[:, 1], predicted_data[:, 1], sequence_length)
        
        self.plot_performance()
    
    def plot_performance(self):
        plt.figure(figsize=(10, 6))
        plt.plot(sequence_lengths, self.mape_results, label=f'{self.ticker} - Open & Close Prices')
        plt.title('Model Performance for Different Sequence Lengths')
        plt.xlabel('Sequence Length')
        plt.ylabel('MAPE (%)')
        plt.legend()
        plt.show()

ticker = 'TSLA'
start_date = '2021-01-01'
end_date = '2021-12-31'

sequence_lengths = [5, 10, 15, 20, 25, 30]

stock_predictor = StockPredictionLSTM(ticker, start_date, end_date)
stock_predictor.run(sequence_lengths)


