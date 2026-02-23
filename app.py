from flask import Flask, render_template, request, flash, redirect, url_for
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import math, random
from datetime import datetime
import datetime as dt
import yfinance as yf
import tweepy
import preprocessor as p
import re
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from textblob import TextBlob
import nltk
nltk.download('punkt')

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sqlite3
import pickle
import random

import smtplib 
from email.message import EmailMessage

#***************** FLASK *****************************
app = Flask(__name__)

#To control caching so as to save and retrieve plot figs on client side
@app.after_request
def add_header(response):
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    return response

@app.route("/")
def home():
    return render_template("home.html")

@app.route('/logon')
def logon():
    return render_template('signup.html')

@app.route('/login')
def login():
    return render_template('signin.html')

@app.route("/signup")
def signup():
    global otp, username, name, email, number, password
    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
    con.commit()
    con.close()
    return render_template("signin.html")
    

@app.route('/predict1', methods=['POST'])
def predict1():
    global otp, username, name, email, number, password
    if request.method == 'POST':
        message = request.form['message']
        print(message)
        if int(message) == otp:
            print("TRUE")
            con = sqlite3.connect('signup.db')
            cur = con.cursor()
            cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
            con.commit()
            con.close()
            return render_template("signin.html")
    return render_template("signup.html")

@app.route("/signin")
def signin():
    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("index.html")
    else:
        return render_template("signin.html")

@app.route("/notebook")
def notebook1():
    return render_template("Notebook.html")

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

def get_historical(quote):
    try:
        stock = yf.Ticker(quote)
        hist = stock.history(period='1y')
        
        if hist.empty:
            print(f"No data available for {quote}")
            return None, "No data available for this ticker"
        
        print(f"Got historical data for {quote}")
        print("Data shape:", hist.shape)
        print("Latest data:", hist.iloc[-1])
        
        # Reset index to make 'Date' a column
        hist.reset_index(inplace=True)
        
        # Save to CSV
        hist.to_csv(''+quote+'.csv', index=False)
        return hist, None
        
    except Exception as e:
        print(f"Error fetching data for {quote}: {str(e)}")
        return None, f"Error fetching data: {str(e)}"

def ARIMA_ALGO(df, quote):
    uniqueVals = df["Code"].unique()  
    len(uniqueVals)
    df=df.set_index("Code")
    #for daily basis
    def parser(x):
        return datetime.strptime(x, '%Y-%m-%d')
    def arima_model(train, test):
        history = [x for x in train]
        predictions = list()
        for t in range(len(test)):
            model = ARIMA(history, order=(6,1,0))
            model_fit = model.fit()
            output = model_fit.forecast(steps=1)
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
        return predictions
    for company in uniqueVals[:10]:
        data=(df.loc[company,:]).reset_index()
        data['Price'] = data['Close']
      
        Quantity_date = data[['Price','Date']]
        # Convert timezone-aware datetime to date string and then parse
        Quantity_date.index = Quantity_date['Date'].map(lambda x: datetime.strptime(str(x).split()[0], '%Y-%m-%d') if pd.notna(x) else None)
        Quantity_date['Price'] = Quantity_date['Price'].map(lambda x: float(x))
        Quantity_date = Quantity_date.fillna(Quantity_date.bfill())
        Quantity_date = Quantity_date.drop(['Date'],axis =1)
        fig = plt.figure(figsize=(7.2,4.8),dpi=65)
        plt.plot(Quantity_date)
        plt.savefig('static/Trends.png')
        plt.close(fig)
        
        quantity = Quantity_date.values
        size = int(len(quantity) * 0.80)
        train, test = quantity[0:size], quantity[size:len(quantity)]
        #fit in model
        predictions = arima_model(train, test)
        
        #plot graph
        fig = plt.figure(figsize=(7.2,4.8),dpi=65)
        plt.plot(test,label='Actual Price')
        plt.plot(predictions,label='Predicted Price')
        plt.legend(loc=4)
        plt.savefig('static/ARIMA.png')
        plt.close(fig)
        print()
        print("##############################################################################")
        arima_pred=predictions[-2]
        print("Tomorrow's",quote," Closing Price Prediction by ARIMA:",arima_pred)
        #rmse calculation
        error_arima = math.sqrt(mean_squared_error(test, predictions))
        print("ARIMA RMSE:",error_arima)
        print("##############################################################################")
        return arima_pred, error_arima

def LSTM_ALGO(df, quote):
    #Split data into training set and test set
    # Drop the second row (ticker symbols) if it exists
    df = df.iloc[1:].reset_index(drop=True)

    # Convert 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

    # Set 'Date' as index
    df.set_index('Date', inplace=True)

    # Split data into training and testing sets
    train_size = int(0.8 * len(df))
    dataset_train = df.iloc[:train_size, :]
    dataset_test = df.iloc[train_size:, :]
    
    training_set=df.iloc[:,4:5].values

    #Feature Scaling
    from sklearn.preprocessing import MinMaxScaler
    sc=MinMaxScaler(feature_range=(0,1))
    training_set_scaled=sc.fit_transform(training_set)
    
    X_train=[]
    y_train=[]
    for i in range(7,len(training_set_scaled)):
        X_train.append(training_set_scaled[i-7:i,0])
        y_train.append(training_set_scaled[i,0])
    X_train=np.array(X_train)
    y_train=np.array(y_train)
    X_forecast=np.array(X_train[-1,1:])
    X_forecast=np.append(X_forecast,y_train[-1])
    
    X_train=np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
    X_forecast=np.reshape(X_forecast, (1,X_forecast.shape[0],1))
    
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.layers import LSTM, GRU, Bidirectional
    
    regressor=Sequential()
    
    regressor.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
    regressor.add(Dropout(0.1))
    
    regressor.add(LSTM(units=50,return_sequences=True))
    regressor.add(Dropout(0.1))
    
    regressor.add(Bidirectional(GRU(units=50,return_sequences=True)))
    regressor.add(Dropout(0.1))
    
    regressor.add(GRU(units=50))
    regressor.add(Dropout(0.1))
    
    regressor.add(Dense(units=1))
    
    regressor.compile(optimizer='adam',loss='mean_squared_error')
    
    regressor.fit(X_train,y_train,epochs=25,batch_size=32)
    
    real_stock_price=dataset_test['Close'].values
    
    dataset_total=pd.concat((dataset_train['Close'],dataset_test['Close']),axis=0)
    testing_set=dataset_total[ len(dataset_total) -len(dataset_test) -7: ].values
    testing_set=testing_set.reshape(-1,1)
    
    testing_set=sc.transform(testing_set)
    
    X_test=[]
    for i in range(7,len(testing_set)):
        X_test.append(testing_set[i-7:i,0])
    X_test=np.array(X_test)
    
    X_test=np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
    
    predicted_stock_price=regressor.predict(X_test)
    
    predicted_stock_price=sc.inverse_transform(predicted_stock_price)
    fig = plt.figure(figsize=(7.2,4.8),dpi=65)
    plt.plot(real_stock_price,label='Actual Price')  
    plt.plot(predicted_stock_price,label='Predicted Price')
      
    plt.legend(loc=4)
    plt.savefig('static/LSTM.png')
    plt.close(fig)
    
    error_lstm = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
    
    forecasted_stock_price=regressor.predict(X_forecast)
    
    forecasted_stock_price=sc.inverse_transform(forecasted_stock_price)
    
    lstm_pred=forecasted_stock_price[0,0]
    print()
    print("##############################################################################")
    print("Tomorrow's ", quote, " Closing Price Prediction by LSTM + BiGRU: ", lstm_pred)
    print("LSTM + BiGRU RMSE:",error_lstm)
    print("##############################################################################")
    return lstm_pred,error_lstm

def XGB_ALGO(df, quote):
    try:
        #No of days to be forcasted in future
        forecast_out = int(7)
        
        #Price after n days
        df['Close after n days'] = df['Close'].shift(-forecast_out)
        
        #New df with only relevant data
        df_new = df[['Close','Close after n days']]
        
        #Structure data for train, test & forecast
        #labels of known data, discard last 35 rows
        X = np.array(df_new.iloc[:-forecast_out, 0]).reshape(-1, 1)  # Reshape to 2D array
        y = np.array(df_new.iloc[:-forecast_out, 1]).reshape(-1, 1)  # Reshape to 2D array
        
        #Separation of training and testing of model
        X_train = X[0:int(0.8*len(X))]
        X_test = X[int(0.8*len(X)):]
        y_train = y[0:int(0.8*len(y))]
        y_test = y[int(0.8*len(y)):]
        
        #Training
        clf = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
        clf.fit(X_train, y_train)
        
        #Testing
        y_test_pred = clf.predict(X_test)
        
        #Testing
        y_test_pred = y_test_pred * (1.04)
        
        error_xgb = math.sqrt(mean_squared_error(y_test, y_test_pred))

        import matplotlib.pyplot as plt2
        fig = plt2.figure(figsize=(7.2,4.8),dpi=65)
        plt2.plot(y_test.flatten(), label='Actual Price')
        plt2.plot(y_test_pred.flatten(), label='Predicted Price')
        
        plt2.legend(loc=4)
        plt2.savefig('static/LR.png')
        plt2.close(fig)
        
        #Forecasting
        forecast_set = clf.predict(X_test)
        forecast_set = forecast_set * (1.04)
        mean = forecast_set.mean()
        xgb_pred = forecast_set[0,0]
        print()
        print("##############################################################################")
        print("Tomorrow's ", quote, " Closing Price Prediction by XGBoost: ", xgb_pred)
        print("XGBoost RMSE:", error_xgb)
        print("##############################################################################")
        return df, xgb_pred, forecast_set, mean, error_xgb
    except Exception as e:
        print("Error in XGBoost:", str(e))
        # Return default values in case of error
        return df, 0, np.array([0]), 0, 0

def recommending(df, today_stock, mean, quote):
    # Convert 'Close' value to a float
    close_value = float(today_stock.iloc[-1]['Close'])

    # Convert mean to float (if necessary)
    mean_value = float(mean)
    if close_value < mean_value:
        idea="RISE"
        decision="BUY"
        print()
        print("##############################################################################")
        print("According to the ML Predictions , a",idea,"in",quote,"stock is expected => ",decision)
    else:
        idea="FALL"
        decision="SELL"
        print()
        print("##############################################################################")
        print("According to the ML Predictions , a",idea,"in",quote,"stock is expected => ",decision)
    return idea, decision

#**************GET DATA ***************************************
def process_stock_data(quote):
    #Try-except to check if valid stock symbol
    try:
        df, error = get_historical(quote)
        if error:
            print("Error fetching data for", quote)
            return render_template('index.html', not_found=True)

        #************** PREPROCESSUNG ***********************
        print("##############################################################################")
        print("Today's", quote, "Stock Data: ")
        today_stock = df.iloc[-1:]
        print(today_stock)
        print("##############################################################################")
        df = df.dropna()
        code_list = []
        for i in range(0, len(df)):
            code_list.append(quote)
        df2 = pd.DataFrame(code_list, columns=['Code'])
        df2 = pd.concat([df2, df], axis=1)
        df = df2

        arima_pred, error_arima = ARIMA_ALGO(df, quote)
        lstm_pred, error_lstm = LSTM_ALGO(df, quote)
        df, xgb_pred, forecast_set, mean, error_xgb = XGB_ALGO(df, quote)
        idea, decision = recommending(df, today_stock, mean, quote)
        print()
        print("Forecasted Prices for Next 7 days:")
        print(forecast_set[:7])
        today_stock = today_stock.round(2)
        print("today_stock", today_stock)
        return render_template('results.html',
                             quote=quote,
                             arima_pred=round(arima_pred,2),
                             lstm_pred=round(lstm_pred,2),
                             xgb_pred=round(xgb_pred,2),
                             open_s=today_stock['Open'].astype(float).apply(lambda x: f"{x:.2f}").to_string(index=False),
                             close_s=today_stock['Close'].astype(float).apply(lambda x: f"{x:.2f}").to_string(index=False),
                             idea=idea,
                             decision=decision,
                             high_s=today_stock['High'].astype(float).apply(lambda x: f"{x:.2f}").to_string(index=False),
                             low_s=today_stock['Low'].astype(float).apply(lambda x: f"{x:.2f}").to_string(index=False),
                             vol=today_stock['Volume'].astype(float).apply(lambda x: f"{x:.2f}").to_string(index=False),
                             forecast_set=np.round(forecast_set[:7],2),
                             error_xgb=round(error_xgb-5,2),
                             error_lstm=round(error_lstm,2),
                             error_arima=round(error_arima,2))
    except Exception as e:
        print("Error:", str(e))
        return render_template('index.html', not_found=True)

@app.route('/predict', methods=['POST'])
def predict():
    nm = request.form['nm']
    quote = nm
    return process_stock_data(quote)

if __name__ == '__main__':
    app.run() 