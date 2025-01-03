import streamlit as st
import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from pmdarima import auto_arima
import matplotlib.pyplot as plt

# ARIMA Forecasting Function
def forecast_with_arima(data, horizon, order, scaler):
    # Scale the data
    scaled_data = scaler.transform(data.values.reshape(-1, 1)).flatten()

    # Train ARIMA
    model = ARIMA(scaled_data, order=order)
    model_fit = model.fit()

    # Generate forecasts
    forecast_scaled = model_fit.forecast(steps=horizon)

    # Inverse scaling
    forecast_unscaled = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1))
    return forecast_unscaled.flatten()

# Streamlit App
st.title("Stock Prediction Chatbot")
st.write("This chatbot predicts stock prices using ARIMA.")

# Input
ticker = st.text_input("Enter Stock Ticker (e.g., ^NSEI):", "^NSEI")
horizon = st.number_input("Enter Prediction Horizon (days):", min_value=1, max_value=30, value=7)

if st.button("Predict with ARIMA"):
    # Fetch historical data
    data = yf.download(ticker, start="2010-01-01", end="2023-12-31")['Close'].dropna()

    if len(data) > 30:
        st.write("Preparing data...")

        # Scale data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

        # Determine ARIMA parameters using auto_arima
        st.write("Determining optimal ARIMA parameters...")
        auto_model = auto_arima(
            data,
            seasonal=True,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            max_order=5,
            trace=True
        )
        arima_order = auto_model.order
        st.write(f"Optimal ARIMA order determined: {arima_order}")

        # Forecast
        st.write("Generating forecast...")
        predictions_arima = forecast_with_arima(data, horizon, order=arima_order, scaler=scaler)

        # Create future dates
        future_dates = pd.date_range(data.index[-1], periods=horizon + 1, freq='B')[1:]

        # Plotting
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data[-50:], label="Historical Prices", color="blue")
        ax.plot(future_dates, predictions_arima, label="ARIMA Forecast", color="red", linestyle="--")
        ax.legend()
        ax.set_title(f"Stock Price Prediction for {ticker}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        st.pyplot(fig)
    else:
        st.write("Not enough data for ARIMA prediction. Ensure at least 30 data points.")
