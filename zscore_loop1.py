# NovaInv September 28, 2021
# Loop through list of assets and backtest based on lowest
# zscores calculated from benchmark

import pandas as pd
import talib
import yfinance as yf
import numpy as np
from datetime import date, timedelta, datetime
from statistics import median
import matplotlib.pyplot as plt 
from math import floor

def benchmark(bench,end,now):
	# performance of benchmark
	data = yf.download(bench, start=end, end=now,progress=False)
	spy = data['Close'].values
	performance = ((spy[-1]/spy[0])-1)*100
	#print(f"{bench} Returned: {performance}")
	return performance

def zscore(timeseries,bench,lookback):
	# calculation of zscore based upon benchmark
	diff = np.log(timeseries) - np.log(bench)
	#mean = talib.SMA(diff,timeperiod=lookback)
	mean = median(diff[-lookback:])
	dev = talib.STDDEV(diff, timeperiod=lookback)
	z = (diff-mean)/dev
	return z[-1]

def performance(df):
	try:
		start = df.iloc[0]
		end = df[-1:].values[0]
		return (end-start)*100/start
	except:
		return 0 

# grab constituent list from file
head_df = pd.read_csv("Datasets/dija_new.csv")
stock_list = head_df['Symbol'].tolist()
#length = 31
portfolio_size = 2 # how many stocks to trade at one time
holding_period = 5 # how long to hold
benchtest = 'DIA'


star = datetime(2020,10,4)
#star = datetime(2020,4,2)
edd = datetime(2022,3,2)
start_date = star.strftime("%Y-%m-%d")
end_date = edd.strftime("%Y-%m-%d")
df = yf.download(stock_list, start=start_date, end=end_date,progress=False) # get data from yahoo finance
df.ffill(inplace=True) # foward fill nan values


spy = yf.download(benchtest, start=start_date,end=end_date,progress=False)
spy.ffill(inplace=True)

for length in range(10,70):
	# loop through varying lookback ranges
	port_list = [0]
	bench_list = [0]
	range_length = floor((len(df)-length)/holding_period)
	for i in range(range_length):
		# step through backtest

		# t = star + timedelta(days=7*i)
		# t7 = t+timedelta(days=7)
		# t35 = t-timedelta(days=length)
		# start_date = t35.strftime("%Y-%m-%d")
		# end_date = t.strftime("%Y-%m-%d")
		# now_date = t7.strftime("%Y-%m-%d")

		rec_list = []
		perf_list = []

		
		# this = df['Adj Close']['AAPL'][-5:]
		# that = df['Adj Close']['AAPL'][-length-5:-5]

		for ticker in stock_list:
			# loop through each stock and caclulate zscore and performance over holding period
			
			fit = zscore(df['Adj Close'][ticker][i*holding_period:length+(i*holding_period)],spy['Adj Close'][i*holding_period:length+(i*holding_period)],length)
			rec_list.append(fit)
			perf_list.append(performance(df['Adj Close'][ticker][length+(i*holding_period):length+holding_period+(i*holding_period)]))
		# 	if ticker == 'AAPL':	
		# 		print(len(df['Adj Close'][ticker][i*holding_period:length+(i*holding_period)]))
		# print(len(spy['Adj Close'][length+(i*holding_period):length+holding_period+(i*holding_period)]))
		# perf_df = yf.download(stock_list, start=end_date, end=now_date,progress=False)
		# perf_df.ffill(inplace=True)
		#print(perf_df)

		#Performance tracking
		
		#print(df['Adj Close'][ticker][-5:])

		#print(i)
		# add results to dataframe
		result = pd.DataFrame(list(zip(stock_list,rec_list,perf_list)), columns=['Symbol','Fit','Perf'])
		result.sort_values(by=['Fit'], inplace=True, ascending=True)
		port_list.append(result['Perf'].head(portfolio_size).mean())
		bench_list.append(performance(spy['Adj Close'][length+(i*holding_period):length+holding_period+(i*holding_period)]))

	# Find performance of strategy and compare to benchmark
	final = np.cumsum(np.array(port_list))
	final_bench = np.cumsum(np.array(bench_list))
	outperf = final[-1]-final_bench[-1]
	print(f"\n{length}:     Outperformance: {outperf}")
	print(final[-1])
	print(final_bench[-1])

# print(result.head(30))
# print(result['Perf'].head(10).mean())
# benchmark(benchtest)
# print(result['Perf'].tail(10).mean())
# print(result.tail(10))

# final = np.cumsum(np.array(port_list))
# final_bench = np.cumsum(np.array(bench_list))
# print(final[-1])
# print(inal_bench[-1])

# plot performance
# plt.plot(range(len(port_list)),final, label = "Portfolio")
# plt.plot(range(len(bench_list)), final_bench, label = "Benchmark")
# plt.legend()
# plt.show()