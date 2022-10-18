# NovaInv April 30, 2021
# Calculate portfolio weightings and performances based
# on Universal Portfolio algorithims
# Reference: https://github.com/Marigold/universal-portfolios

import numpy as np
import pandas as pd 
import yfinance as yf
from datetime import date, timedelta
import matplotlib.pyplot as plt 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def compounding(perff,leverage=1):
	# calculate returns compounded
	comp = 1
	for i in range(len(perff)):
		comp = comp*(perff[i]*leverage/100+1)
	return (comp-1)*100


from universal.algos import *

plt.rcParams["figure.figsize"] = (12,6)

start_date = '2019-11-12'
end_date = '2022-09-03'
#start_date = '2021-04-01'
#end_date = date.today() + timedelta(days=1)

# grab index constituents from file or use list
head_df = pd.read_csv("Datasets/nasdaq100.csv")
stock_list = head_df['Symbol'].tolist()
useAlgos = True # use all aglorithims? 
#stock_list = ['AAPL','MSFT']
#stock_list = ['TQQQ','SQQQ']
#stock_list = ['SMH','SPY','XLB','XLE','XLF','XLI','XLP','XLRE','XLU','XLV','XLY']
#stock_list = ['AAPL','AAL','AMD','BAC','F','META','NIO','NVDA','SPY','TSLA','XLF']
#stock_list = ['AAPL','GOOG','MSFT','AMZN','BRK-B','META','JPM','JNJ','BAC','WMT','V','PFE','HD','DIS','NVDA']

# get data from yahoo finance
df = yf.download(stock_list, start=start_date, end=end_date,progress=False)['Adj Close']
print(df.isnull().values.any()) # any nans?



# find optimal parameter
# for i in range(5,31):
# 	algo2 = BNN(l=i)
# 	result2 = algo2.run(df['Adj Close'])
# 	print(f"\nWindow of {i}:")
# 	print(result2.summary())

def all_algos(df):
	# backtest each algorithim
	# *probably a simpler way to do this
	# set algo parameters
	#algo1 = Anticor()
	algo2 = BAH()
	algo3 = BNN() #check if l=1
	algo4 = CORN()
	algo5 = CRP()
	#algo6 = CWMR() #check
	algo7 = EG() #mixed
	algo8 = OLMAR()
	algo9 = ONS() #MIXED
	algo10 = PAMR() #check
	algo11 = RMR()
	algo12 = UP() #mixed
	print("Algos Loaded")
	# run
	#result1 = algo1.run(df['Adj Close'])
	result2 = algo2.run(df)
	result3 = algo3.run(df)
	result4 = algo4.run(df)
	result5 = algo5.run(df)
	#result6 = algo6.run(df)
	result7 = algo7.run(df)
	result8 = algo8.run(df)
	result9 = algo9.run(df)
	result10 = algo10.run(df)
	result11 = algo11.run(df)
	result12 = algo12.run(df)
	print("Algos Tested")

	#print("Anticorrelation:")
	#print(result1.summary())
	print("\nBuy and Hold:")
	print(result2.summary())
	print("\nNearest Neighbor Log-Optimal:")
	print(result3.summary())
	print("\nCorrelation-driven:")
	print(result4.summary())
	print("\nConstant Rebalanced:")
	print(result5.summary())
	print("\nConfidence Weighted Mean Reversion:")
	#print(result6.summary())
	print("\nExponetial Gradient:")
	print(result7.summary())
	print("\nOnline Moving Average Mean Reversion:")
	print(result8.summary())
	print("\nOnline Newton Step:")
	print(result9.summary())
	print("\nPassive Aggressive Mean Reversion:")
	print(result10.summary())
	print("\nRobust Median Reversion:")
	print(result11.summary())
	print("\nUniversal Portfolio:")
	print(result12.summary())

if useAlgos:
	all_algos(df)
else:
	# use specific algorithim
	algo = ONS()
	result = algo.run(df)
	print(result.summary())
	#print(result.B)
	result.plot(logy=False,assets=False,weights=False,ucrp=True) # plot performance of algo
	plt.show()

	print(f"\nYesterday: {result.B.index[-2]}") # print second to last weightings
	for i in range(len(stock_list)):
		weight = result.B.iloc[-2][i]
		if weight > 0.0001:
			print(f"{stock_list[i]}: {weight}")

	print(f"\nToday: {result.B.index[-1]}") # print last weightings
	for i in range(len(stock_list)):
		weight = result.B.iloc[-1][i]
		if weight > 0.0001:
			print(f"{stock_list[i]}: {weight}")
		else:
			print(f"{stock_list[i]}: {0}")



