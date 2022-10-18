# NovaInv November 30, 2020
# Compute trending or mean reversion statistics for 
# intraday prices of stock lists

import talib
import pandas as pd
import numpy as np
from numpy import log, polyfit, sqrt, std, subtract
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import date, timedelta
pd.set_option('display.max_columns', None)

start_date = date(2022,8,4)#"2022-08-08"
# today = date.today() + timedelta(days=1)
# end_date = today.strftime("%Y-%m-%d")
end_date = start_date + timedelta(days=2) #"2022-08-07"
start = start_date.strftime("%Y-%m-%d")
end = end_date.strftime("%Y-%m-%d")
linreg_lookback = 24

head_df = pd.read_csv("Datasets/nasdaq100.csv") #grab list of nasdaq100 stocks
stock_list = head_df['Symbol'].tolist()

def hurst(ts):
	# calculate the hurst exponent to determine trending or mean reversion
	ts = ts if not isinstance(ts, pd.Series) else ts.to_list()
	lags = range(2,100)
	tau = [sqrt(std(subtract(ts[lag:],ts[:-lag]))) for lag in lags]
	poly = polyfit(log(lags),log(tau),1)
	return poly[0]*2.0

def linreg_rank(df):
	# calculate linear regression and r^2
	# multiply together to find the best fitting up trend
	# loghigh = np.log10(df['High']) #Use log prices or no
	# loglow = np.log10(df['Low'])
	# logclose = np.log10(df['Close'])
	loghigh = np.array(df['High'])
	loglow = np.array(df['Low'])
	logclose = np.array(df['Close'])
	try:
		r_squared = pow(talib.CORREL(loghigh,loglow,timeperiod=len(df)),2)
		slope = talib.LINEARREG_SLOPE(logclose, timeperiod=len(df))
		#print(r_squared[-1] * slope[-1])
		return r_squared[-1] * slope[-1]
	except:
		return 0
def performance(df):
	# get performance of timeseries
	# needs improving
	try:
		start_close = df.iloc[0]['Adj Close']
		last_close = df[-1:]['Adj Close'].values[0]
		return (last_close-start_close)*100/start_close
	except:
		return 0

def zscore(data, window):
	# calculate zscore for give timeseries and length
    rolling_mean = data['Close'].rolling(window).mean()
    rolling_std = data['Close'].rolling(window).std()
    df['zscore'] = (df['Close']-rolling_mean)/rolling_std
###################
def logic(df,window):
	# logic to determine if high in second period is greater than first period
	# needs addition: doesn't account for drawdown
	try:
		late_high = df[window:]['Close'].max()
		early_high = df[:window]['Close'].max()
		if late_high>early_high:
			return True
		else:
			return False
	except:
		return False

def logic_percent(df,window):
	# calculate percent difference of high in second period to high in first period
	try:
		late_high = df[window:]['Close'].max()
		early_high = df[:window]['Close'].max()
		return (late_high-early_high)*100/early_high
	except:
		return 0
####Test df########
# temp_df = yf.download("SPY", start=start_date, end=end_date ,interval="5m",progress=False)
# print(temp_df[:3])
##################
#print(performance(temp_df[round(len(temp_df)/2):]))

leader = [0]
lagger = [0]
total = [0]
# start_date = date(2022,7,26)#"2022-08-08"
# for i in range(1,18):


# 	if i != 1:
# 		start_date = start_date + timedelta(days=1)
# 	end_date = start_date + timedelta(days=2)
# 	start = start_date.strftime("%Y-%m-%d")
# 	end = end_date.strftime("%Y-%m-%d")

# list to store and retrieve variables
linreg_list = []
perf_list = []
hurst_list = []
logic_list = []
#print(yf.download('AAPL', start=start_date, end=end_date ,interval="1m",progress=False).head())

# grab stocks using yahoo finance
print(start)
df = yf.download(stock_list, start=start, end=end ,interval="5m",progress=False,group_by='ticker')
df.bfill(inplace=True)
print(len(df))
print(df.index[77])
print(len(df['AAPL'].iloc[77:78]))

# loop through each stock and store statistics
for ticker in stock_list:
	#df = yf.download(ticker, period="1d",interval="5m",progress=False)
	
	#print(df)
	#print(len(df[:linreg_lookback]))
	#change = (df['Close'][0]-df['Open'][0])*100/df['Open'][0]
	linreg_list.append(linreg_rank(df[ticker].iloc[42:77]))
	perf_list.append(performance(df[ticker].iloc[77:79]))
	#perf_list.append(change)
	#zscore(df,20)
	#df.dropna(inplace=True)
	#hurst_list.append(hurst(df['zscore']))

# store variables in dataframe and sort by different metrics
#print(df[:60])
head_df = pd.DataFrame({'Ticker':stock_list,'Slope':linreg_list,'Perf':perf_list})
#head_df['hurst'] = hurst_list
head_df.sort_values(by=['Slope'], inplace=True, ascending=False)
leaderRet = head_df['Perf'].head(10).mean()
laggerRet = head_df['Perf'].tail(10).mean()
print(f"Leader Return: {leaderRet}")
print(f"Lagger Return: {laggerRet}")
print(head_df.head(15))
# 	lagger.append(laggerRet)
# 	leader.append(leaderRet)
# 	total.append((laggerRet+leaderRet)/2)

# plot leading and lagging stocks and respective performances
# length = range(len(lagger))
# plt.plot(range(len(lagger)),np.cumsum(np.array(lagger)),label='Lagger')
# plt.plot(range(len(lagger)),np.cumsum(np.array(leader)),label='Leader')
# plt.plot(range(len(lagger)),np.cumsum(np.array(total)),label='Total')
# plt.legend(loc="upper left")
# plt.show()
#print(head_df['Slope'].head(15).mean())
