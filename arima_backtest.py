# NovaInv December 14, 2020
# Backtest simple ARIMA model using one-step predictions
# References: https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/

import numpy as np 
import pmdarima as pm
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA 
from statsmodels.tsa.forecasting.stl import STLForecast
import pandas as pd 
import yfinance as yf
import math
'''
This strategy is to buy at open and have take profit set at fc
'''
auto_arima_choice = False #use autoarima function or predifined variables
lookback = 250 #lookback range for model at each step
start_date = '2016-04-18'
end_date = '2022-01-01'

# Use just SPY or multiple large stocks
data = ["SPY"] #,"TSLA","AMZN","AMD","FB","SNAP","BAC","AAL","T","PFE","GME","CRM","JPM","DIS","PYPL","WMT","NKE","C","XOM","MSFT"]
head_df = pd.DataFrame(data, columns=['Symbols'])
stock_list = head_df['Symbols'].to_list()
# cum_list = []
# rate_list = []
# av_list = []

# loop through each ticker symbol
for ticker in stock_list:
	df = yf.download(ticker, start=start_date,end=end_date,progress=False) # grab data from yahoo finance
	df.drop(columns=['Low', 'Volume'],inplace=True)
	#df['log return'] = np.log(df['Close']).diff()
	#df['log return'] = df['Adj Close'].pct_change()
	df['log return'] = (df['Open']/df['Close'].shift(1)-1)*100 # calculate log returns of overnight gap
	df.dropna(inplace=True)
	print("Collected Data.")
	#for lookback in range(5,101):
	# list to store and retrieve variables
	performance_list = []
	hit_list = []
	#print(len(df)-lookback)

	# backtest over each step
	for i in range(len(df)-lookback):
		#print(df[i:lookback+i])
		open_ = df.iloc[lookback+i-1]['Close']
		close = df.iloc[lookback+i]['Open']
		#open_ = df.iloc[lookback+i]['Open']
		#high = df.iloc[lookback+i]['High']
		average = df[i:lookback+i]['log return'].mean()
		this = (close-open_)*100/open_
		# use auto arima functoin
		if auto_arima_choice:
			model = pm.auto_arima(df[i:lookback+i]['log return'], start_p=1, start_q=1,
		                      test='adf',       # use adftest to find optimal 'd'
		                      max_p=4, max_q=4, # maximum p and q
		                      m=1,              # frequency of series
		                      d=None,           # let model determine 'd'
		                      seasonal=False,   # No Seasonality
		                      start_P=0, 
		                      D=None, 
		                      trace=False,
		                      error_action='warn',  
		                      suppress_warnings=True, 
		                      stepwise=True)
			n_periods = 1
			prediction = model.predict(n_periods=n_periods, return_conf_int=False) # predict one step ahead

		else: # use predefined variables
			model = pm.ARIMA(order=(1,0,0),suppress_warnings=True)
			model.fit(df[i:lookback+i]['log return'])
			#print(model.summary())
			n_periods = 1
			prediction = model.predict(n_periods=n_periods, return_conf_int=False) # predict one step ahead
			#print(prediction)

		if prediction>(average) and average>0:
			# is prediction higher?
			if close>open_:
				# is the actual gap up or down?
				performance_list.append(this)
				hit_list.append(1)
			if close<=open_:
				performance_list.append(this)
				hit_list.append(0)
		'''
		fc = math.exp(prediction) * prev_close
		if fc>prev_close:
			if high>=fc and open_<fc:
				performance_list.append((fc-open_)*100/open_)
				hit_list.append(1)
			if high<fc:
				nonadj_close = df.iloc[lookback+i]['Close']
				performance_list.append((nonadj_close-open_)*100/open_)
				hit_list.append(0)
				#print(df.iloc[lookback+i])'''
	# performance statistics
	perf_arr = np.array(performance_list)
	totalReturn = np.sum(perf_arr)
	#cum_list.append(totalReturn)
	avReturn = np.mean(perf_arr)
	#av_list.append(avReturn)
	minReturn = np.min(perf_arr)
	maxReturn = np.max(perf_arr)
	winRate = np.mean(hit_list)
	#rate_list.append(winRate)

	# print preformance statistics
	print(f"\n{ticker} with lookback of {lookback}:")
	print(f"Cumulative Return: {totalReturn}")
	print(f"Average Daily Return: {avReturn}")
	print(f"Win Percentage: {winRate}")
	print(f"\n{len(perf_arr)} Days traded out of {len(df)-lookback} total Days")
	print(f"Worst Day: {minReturn}")
	print(f"Best Day: {maxReturn}")

# head_df['Total Return'] = cum_list
# head_df['Avg Return'] = av_list
# head_df['Win Rate'] = rate_list
# head_df.sort_values(by=['Total Return'],inplace=True,ascending=False)
# print(head_df.head(len(head_df)))

# plot performance
plt.bar(range(len(performance_list)),np.cumsum(np.array(performance_list)))
plt.show()
# start_date = '2020-12-18'
# end_date = '2021-09-30'
# df = yf.download("SPY", start=start_date,end=end_date,progress=True)
# df['gap'] = (df['Open']/df['Close'].shift(1)-1)*100
# print(df['Close'].tail())
# print(df.tail())