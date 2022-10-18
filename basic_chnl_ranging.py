# NovaInv November 19, 2020
# Strictly technical analysis to find trending or 
# ranging stocks from a given list and time period
# References: https://www.youtube.com/watch?v=exGuyBnhN_8&ab_channel=PartTimeLarry

import talib
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import date, timedelta
from statsmodels.tsa.stattools import adfuller

start_date = '2021-08-02'
today = date.today() + timedelta(days=1)
end_date = today.strftime("%Y-%m-%d")
end_date = '2021-11-20'
linreg_lookback = 60
adf_lookback = 60
holding_period = 20
range_lookback = 15
percent_range = 5

linreg_selection = 0

head_df = pd.read_csv("Datasets/SandP.csv")
stock_list = head_df['Symbol'].tolist()

def r_squared(df):
	# calculate r^2
	recents = df[-linreg_lookback:]
	loghigh = np.log(recents['High'])
	loglow = np.log(recents['Low'])
	try:
		r_squared = pow(talib.CORREL(loghigh,loglow,timeperiod=linreg_lookback),2)
		return r_squared[-1]
	except:
		return 0

def linreg_rank(df):
	# calculate linear regression and r^2 to find the best fit up trend
	recents = df[-linreg_lookback:]
	loghigh = np.log(recents['High'])
	loglow = np.log(recents['Low'])
	logclose = np.log(recents['Close'])
	try:
		r_squared = pow(talib.CORREL(loghigh,loglow,timeperiod=linreg_lookback),2)
		slope = talib.LINEARREG_SLOPE(logclose, timeperiod=linreg_lookback)
		#print(r_squared[-1])
		#print(slope[-1])
		#print(r_squared[-1] * slope[-1])
		return r_squared[-1] * slope[-1]
	except:
		return 0

# def morningstar(df):
# 	try:
# 		num = talib.CDLMORNINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
# 	except:
# 		return False
		
# 	if num[-1] > 0:
# 		return True

def is_consolidating(df, percentage):
	# see if time series is within a given range for a time period
	recent_candlessticks = df[-range_lookback:]
	max_close = recent_candlessticks['Close'].max()
	min_close = recent_candlessticks['Close'].min()
	
	if min_close >= (max_close * (1-percentage/100)):
		return True

def is_breaking_out(df,percentage):
	# see if a time series has broken out of a given range
	try:
		last_close = df[-1:]['Close'].values[0]
	except:
		return False

	if is_consolidating(df[:-1],percentage):
		recent_closes = df[(-range_lookback-1):-1]

		if last_close > recent_closes['High'].max():
			return True
	return False

def performance(df):
	# calculate performance of time series
	# needs improving
	try:
		start = df.iloc[0]['Close']
		end = df[-1:]['Close'].values[0]
		return (end-start)*100/start
	except:
		return 0 

# list to store and retrieve variables
consolidating_list = []
breakout_list = []
mstar_list = []
linreg_list = []
perf_list = []
adf_list = []
correlation_list = []

# iterate over stocks in list by choosen tasks
for ticker in stock_list:
	df = yf.download(ticker, start=start_date, end=end_date,progress=False)
	if linreg_selection == 0:
		if is_consolidating(df,3):
			consolidating_list.append(ticker)
		if is_breaking_out(df, 5):
			breakout_list.append(ticker)
		# if morningstar(df):
		# 	mstar_list.append(ticker)
	elif linreg_selection == 1:
		linreg_list.append(linreg_rank(df[-(linreg_lookback+holding_period):-holding_period]))
		perf_list.append(performance(df[-holding_period-1:]))
		correlation_list.append(r_squared(df[-(linreg_lookback+holding_period):-holding_period]))
	elif linreg_selection == 2:
		try:
			# use Augmented Dickey Fuller test to test for mean reversion
			result = adfuller(df[-(adf_lookback+holding_period):-holding_period]['Close'], autolag='AIC')
			adf_list.append(result[0])
		except:
			adf_list.append(0)
		linreg_list.append(linreg_rank(df[-(linreg_lookback+holding_period):-holding_period]))
		perf_list.append(performance(df[-holding_period-1:]))
	else:
		linreg_list.append(linreg_rank(df[-linreg_lookback:]))
		correlation_list.append(r_squared(df[-linreg_lookback:]))
print(df)

# print results given selected tasks
if linreg_selection == 0:
	# look for consolidated and breaking out stocks
	print('\nThe consolidating stocks are:\n')
	print(consolidating_list)
	print('\nThe breaking out stocks are:\n')
	print(breakout_list)
	print('\nMorning Star Stocks:\n')
	print(mstar_list) 

elif linreg_selection == 1:
	# find trendign sotcks and sort
	head_df['Slope'] = linreg_list
	head_df['Perf'] = perf_list
	head_df['R^2'] = correlation_list
	mask = head_df[ head_df['R^2'] < 0.9699 ].index
	head_df.sort_values(by=['Slope'], inplace=True, ascending=False)
	#print(head_df.head(20))
	print('Highest Slope Returns: ' )
	print(head_df['Perf'].head(20).mean())
	print(head_df.head(20))

	head_df.sort_values(by=['Slope'], inplace=True, ascending=True)
	#print(head_df.head(20))
	print('\nLowest Slope Returns: ')
	print(head_df['Perf'].head(20).mean())

	head_df.drop(mask, inplace=True)
	head_df.sort_values(by=['Slope'], inplace=True, ascending=False)
	print('\n(new) Highest Slope Returns: ' )
	print(head_df['Perf'].head(15).mean())
	print(head_df.head(15))

	head_df.sort_values(by=['Slope'], inplace=True, ascending=True)
	#print(head_df.head(20))
	print('\n(new) Lowest Slope Returns: ')
	print(head_df['Perf'].head(15).mean())

elif linreg_selection == 2:
	# find stocks that are mean reverting based on adf test
	head_df['adf'] = adf_list
	head_df['Perf'] = perf_list
	head_df['Slope'] = linreg_list
	head_df.sort_values(by=['adf'], inplace=True, ascending=False)
	print(head_df.head(20))
	print(head_df['Perf'].head(15).mean())

	head_df.sort_values(by=['Slope'], inplace=True, ascending=False)
	#print(head_df.head(20))
	print(head_df.head(20))
	print('Highest Slope Returns: ' )
	print(head_df['Perf'].head(15).mean())

else:
	# simpler search for trending stocks with best fit
	head_df['Slope'] = linreg_list
	head_df['R^2'] = correlation_list
	head_df.sort_values(by=['Slope'], inplace=True, ascending=False)
	mask = head_df['R^2'] > 0.9699
	print("Treding UP:")
	print(head_df[mask].head(20))

	head_df.sort_values(by=['Slope'], inplace=True, ascending=True)
	print("\nTrending DOWN:")
	print(head_df[mask].head(20))


#mask = (head_df['Slope'] <= 0.02) & (head_df['Slope'] >= 0.013)
#print(head_df[mask])
