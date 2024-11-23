import pandas as pd
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice
from ta.momentum import ROCIndicator

# Sample DataFrame with closing prices (replace 'c' with your actual closing price data)
data = pd.DataFrame({
    'Close': [10, 10.5, 10.8, 11, 10.7, 10.9, 11.2, 11.5, 11.1, 11.3]
})

# Calculate MACD
macd = MACD(close=data['Close'], window_slow=26, window_fast=12, window_sign=9)
data['MACD'] = macd.macd()

# Calculate RSI
rsi = RSIIndicator(close=data['Close'], window=14)
data['RSI'] = rsi.rsi()

# Calculate Bollinger Bands
bb = BollingerBands(close=data['Close'], window=5, window_dev=2)
data['BBANDS_U'] = bb.bollinger_hband()
data['BBANDS_L'] = bb.bollinger_lband()
data['width'] = data['BBANDS_U'] - data['BBANDS_L']

# Note: The `HT_DCPERIOD` function is not directly available in `ta`, so you might need a custom implementation.

# Calculate ROC
roc = ROCIndicator(close=data['Close'], window=12)
data['roc'] = roc.roc()

# Calculate the difference of closing prices
data['diff'] = data['Close'].diff(1)

# Display the DataFrame with all calculated indicators
print(data)
