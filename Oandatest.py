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
    
def check_positions_and_sell(api, account_id, instrument, model_prediction, last_known_price):
    account_info = fetch_account_details(api, account_id)
    
    if account_info:
        # Example: Check if there are existing positions for the instrument
        positions = account_info['account']['positions']
        for position in positions:
            if position['instrument'] == instrument:
                units = int(position['long']['units']) if float(position['long']['units']) > 0 else int(position['short']['units'])
                unrealized_pl = float(position['unrealizedPL'])

                # Example Selling Logic: Close position if there's a profit or if the model predicts a decrease
                if unrealized_pl > 0 or model_prediction < last_known_price:
                    print(f"Selling {units} units of {instrument} due to profit or predicted decrease.")
                    place_sell(api, account_id, instrument, -units)  # Negative units for selling
                break

def place_sell(api, account_id, instrument, units):
    data = {
        "order": {
            "instrument": instrument,
            "units": units,
            "type": "MARKET",
        }
    }
    r = orders.OrderCreate(account_id, data)
    try:
        api.request(r)
    except oandapyV20.exceptions.V20Error as err:
        print("Error: {}".format(err))

def place_order(api, account_id, instrument, units, stop_loss_distance):
    data = {
        "order": {
            "instrument": instrument,
            "units": units,
            "type": "MARKET",
            "stopLossOnFill": {
                "distance": str(stop_loss_distance)
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
        'time': candle['time']
    } for candle in candle_data])

    df['MA_50'] = calculate_moving_average(df['price'], 50)
    df['MA_200'] = calculate_moving_average(df['price'], 200)
    df['RSI'] = calculate_rsi(df['price'], 14)

    df.dropna(inplace=True)  # Drop rows with NaN values
    return df

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

        # Define features and target
        X = data[['MA_50', 'MA_200', 'RSI']]
        y = data['price']

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Train a Random Forest Regressor
        model = RandomForestRegressor(n_estimators=100, random_state=0)
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Compare predictions with actual values
        for pred, actual in zip(predictions[:10], y_test[:10]):
            print(f'Predicted: {pred}, Actual: {actual}')

        predictions = model.predict(X_test)

        # Compare predictions with actual values
        for pred, actual in zip(predictions[:10], y_test[:10]):
            print(f'Predicted: {pred}, Actual: {actual}')

        latest_features = X.iloc[-1].to_frame().transpose()  # Convert the last row to a DataFrame
        next_price_prediction = model.predict(latest_features)[0]
        last_known_price = y.iloc[-1]

        # Check positions and decide on selling
        check_positions_and_sell(api, account_id, instrument, next_price_prediction, last_known_price)

        # Buy logic (as previously implemented)
        if next_price_prediction > last_known_price:
            print(f"Placing Buy Order: Predicted Price: {next_price_prediction}, Last Known Price: {last_known_price}")
            place_order(api, account_id, instrument, 100, 0.001)  # Example: Buying 100 units
        else:
            print("No action taken")

if __name__ == "__main__":
    main()
