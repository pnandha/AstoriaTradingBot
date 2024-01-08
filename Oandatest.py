import oandapyV20
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.accounts as accounts

def fetch_account_details(api, account_id):
    r = accounts.AccountDetails(accountID=account_id)
    try:
        account_info = api.request(r)
        return account_info
    except oandapyV20.exceptions.V20Error as err:
        print("Error: {}".format(err))
        return None
    
def check_positions_and_decide_action(api, account_id, instrument, trading_signal):
    account_info = fetch_account_details(api, account_id)
    if account_info:
        positions = account_info['account']['positions']
        for position in positions:
            if position['instrument'] == instrument:
                units_long = int(position['long']['units'])
                units_short = int(position['short']['units'])

                if units_long > 0 and trading_signal == 'sell':
                    # Close long position
                    return -units_long
                elif units_short > 0 and trading_signal == 'buy':
                    # Close short position
                    return -units_short
                elif units_long == 0 and units_short == 0:
                    # No existing position, follow the trading signal
                    if trading_signal == 'buy':
                        return 100  # Example: Buying 100 units
                    elif trading_signal == 'sell':
                        return -100  # Example: Selling 100 units
    return 0  # Default to no action
    
def place_order(api, account_id, instrument, units, stop_loss, take_profit):
    data = {
        "order": {
            "instrument": instrument,
            "units": units,
            "type": "MARKET",
            "stopLossOnFill": {
                "price": str(stop_loss)
            },
            "takeProfitOnFill": {
                "price": str(take_profit)
            }
        }
    }
    r = orders.OrderCreate(account_id, data)
    try:
        api.request(r)
    except oandapyV20.exceptions.V20Error as err:
        print("Error: {}".format(err))

def calculate_moving_average(prices, window):
    return prices.rolling(window=window).mean()

def calculate_rsi(prices, window):
    delta = prices.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    gain = up.rolling(window=window).mean()
    loss = down.abs().rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def fetch_data(api, instrument, count, granularity):
    params = {
        "count": count,
        "granularity": granularity
    }
    candles = instruments.InstrumentsCandles(instrument=instrument, params=params)
    api.request(candles)
    return candles.response['candles']

def prepare_data(candle_data):
    df = pd.DataFrame([{
        'price': float(candle['mid']['c']),
        'high': float(candle['mid']['h']),
        'low': float(candle['mid']['l']),
        'volume': float(candle['volume']),
        'time': candle['time']
    } for candle in candle_data])

    # Calculate Moving Averages
    df['MA_50'] = calculate_moving_average(df['price'], 50)
    df['MA_200'] = calculate_moving_average(df['price'], 200)

    # Calculate RSI
    df['RSI'] = calculate_rsi(df['price'], 14)

    # Calculate Volume Average
    df['Volume_Avg'] = df['volume'].rolling(window=50).mean()

    # Calculate Volatility (using Average True Range as an example)
    df['ATR'] = calculate_atr(df['high'], df['low'], df['price'], 14)

    df.dropna(inplace=True)  # Drop rows with NaN values
    return df

def evaluate_trading_signal(data_row):
    # Placeholder for complex trading signal evaluation logic
    # Example: Check if price is above MA_200 and RSI is below 70 for a buy signal
    if data_row['MA_200'] < data_row['price'] and data_row['RSI'] < 70:
        return 'buy'
    elif data_row['MA_200'] > data_row['price'] and data_row['RSI'] > 30:
        return 'sell'
    return 'hold'

def calculate_stop_loss_take_profit(price, signal, instrument):
    # Adjust these values as per your risk management strategy
    stop_loss_buffer = 0.001  # Example buffer
    take_profit_buffer = 0.001  # Example buffer

    if instrument in ['EUR_USD', 'GBP_USD', 'AUD_USD']:  # Assuming these are similar in precision
        stop_loss = round(price + stop_loss_buffer if signal == 'sell' else price - stop_loss_buffer, 5)
        take_profit = round(price - take_profit_buffer if signal == 'sell' else price + take_profit_buffer, 5)
    elif instrument == 'USD_CAD':  # Adjust if different precision is needed
        stop_loss = round(price + stop_loss_buffer if signal == 'sell' else price - stop_loss_buffer, 5)
        take_profit = round(price - take_profit_buffer if signal == 'sell' else price + take_profit_buffer, 5)
    # Add more conditions if you have other instruments with different precisions

    return stop_loss, take_profit

def calculate_atr(high, low, close, window):
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    ranges = np.vstack([high_low, high_close, low_close])
    true_range = np.max(ranges, axis=0)
    atr = pd.Series(true_range).rolling(window=window).mean()
    return atr

def main():
    access_token = 'dbf13bc036f3f4d00de4e92c84ec7e44-fbb6589170a6fed95593e632dc70f7a7'
    account_id = '101-004-27786726-001'
    api = API(access_token=access_token)
    instruments_list = ["EUR_USD", "GBP_USD", "AUD_USD", "USD_CAD"]
    for instrument in instruments_list:
    # Fetch historical data
        historical_data = fetch_data(api, instrument, 1500, "H1")

        # Prepare data
        data = prepare_data(historical_data)

        # Define features and target. Now including Volume_Avg and ATR.
        X = data[['MA_50', 'MA_200', 'RSI', 'Volume_Avg', 'ATR']]
        y = data['price']

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Train a Random Forest Regressor
        model = RandomForestRegressor(n_estimators=100, random_state=0)
        model.fit(X_train, y_train)

        # Make predictions and compare with actual values
        predictions = model.predict(X_test)
        for pred, actual in zip(predictions[:10], y_test[:10]):
            print(f'Predicted: {pred}, Actual: {actual}')

        # Use the latest data point for prediction
        latest_data = data.iloc[-1]
        trading_signal = evaluate_trading_signal(latest_data)
        action_units = check_positions_and_decide_action(api, account_id, instrument, trading_signal)

        if action_units != 0:
            stop_loss, take_profit = calculate_stop_loss_take_profit(latest_data['price'], trading_signal, instrument)
            print(f"Placing Order for {instrument}: Units = {action_units}")
            place_order(api, account_id, instrument, action_units, stop_loss, take_profit)
        else:
            print(f"No action taken for {instrument}")
            
        if trading_signal != 'hold':
            stop_loss, take_profit = calculate_stop_loss_take_profit(latest_data['price'], trading_signal, instrument)
            units = 100  # Define your position size logic
            if trading_signal == 'buy':
                print(f"Placing Buy Order for {instrument}")
                place_order(api, account_id, instrument, units, stop_loss, take_profit)
            elif trading_signal == 'sell':
                print(f"Placing Sell Order for {instrument}")
                place_order(api, account_id, instrument, -units, stop_loss, take_profit)
        else: 
            print(f"No action taken for {instrument}")

if __name__ == "__main__":
    main()
