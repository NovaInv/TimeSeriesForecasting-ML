# NovaInv April 21, 2022
# Strategy of asset selection based upon mean and 
# median returns of group. Then selects best, worst, or
# median performing asset.

import pandas as pd
import yfinance as yf
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from statistics import median, mean
from datetime import datetime, timedelta
from math import floor
import time
import random

pd.set_option('display.max_rows', None)

def benchmark(ticker):
	# performance of benchmark
	data = yf.download(ticker, start=start_date, end=end_date,progress=False)
	spy = data['Close'].values
	performance = ((spy[-1]/spy[0])-1)*100
	print(f"{ticker} Returned: {performance}")

def compounding(perff,leverage=1):
	# calculate returns compounded
	comp = 1
	for i in range(len(perff)):
		comp = comp*(perff[i]*leverage/100+1)
	return (comp-1)*100

start_date = '2019-01-01'
end_date = '2022-07-27'

#     Must be in alphabetical order
#stock_list = ['ADA-USD','BNB-USD','BTC-USD','DOGE-USD','ETH-USD','LINK-USD','XRP-USD']
stock_list = ['SMH','SPY','XLB','XLE','XLF','XLI','XLP','XLRE','XLU','XLV','XLY']
#stock_list = ['A','AAPL','AMZN','FB','GOOG','HD','JPM','MSFT','NVDA','SPY','TSLA','TSM','UNH'] #biggest stocks
#stock_list = ['AAPL','AAL','AMD','BAC','F','FB','NIO','NVDA','SPY','TSLA','XLF']
#stock_list = ['AAPL','GOOG','MSFT','AMZN','BRK-B','FB','JPM','JNJ','BAC','WMT','V','PFE','HD','DIS','NVDA']
## don't forget to fillna

# #########################
# #array [packet]  [row within packet (day)]  [specific value from  that day/row]
# #########################
def backtest_loop():
	# backtest loop to find optimal lookback parameter
	whileLoop = True
	while whileLoop:
		# continue retrieving data until no nan values
		df = yf.download(stock_list, start=start_date, end=end_date,progress=False)
		whileLoop = df.isnull().values.any()
		if whileLoop:
			print("Error")
			time.sleep(15)
	df.bfill(axis='rows',inplace=True)
	df.dropna(inplace=True)
	#print(df.head(15))

	print("Collected Tickers")

	# list to store and retrieve variables
	longperf1 = []
	longperf2 = []
	longperf3 = []
	longperf4 = []
	overall_list = []
	overallAcc = []
	for seq_len in range(10,21):

		sequential_data = []  
		prev_days = deque(maxlen=seq_len+2) 

		for i in df['Adj Close'].values: #df['Adj Close'].values:  # iterate over the values
			prev_days.append([n for n in i]) 
			if len(prev_days) == (seq_len+2):  
				sequential_data.append(np.array(prev_days))
		seq = np.array(sequential_data) # create 3 dimensional array

		performance = []
		perf1 = []
		perf2 = []
		perf3 = []
		perf4 = []
		for packet in range(seq.shape[0]):
			
			ratio_list = []
			long_list = []
			for item in range(seq.shape[2]):
				ratio = seq[packet][seq_len][item]/seq[packet][seq_len-1][item]
				ratio_list.append(ratio)

			best = max(ratio_list)
			worst = min(ratio_list)
			meen = mean(ratio_list)
			medi = median(ratio_list)
			best_locale = ratio_list.index(best)
			worst_locale = ratio_list.index(worst)
			med_locale = ratio_list.index(medi)

			for item in range(seq.shape[2]):
				ratio = seq[packet][seq_len][item]/seq[packet][0][item]
				long_list.append(ratio)

			lookback_mean = mean(long_list)
			lookback_medi = median(long_list)
			lookback_best = max(long_list)
			lookback_worst = min(long_list)
			lookbackbest_locale = long_list.index(lookback_best)
			lookbackmed_locale = long_list.index(lookback_medi)
			lookbackworst_locale = long_list.index(lookback_worst)
			dirac = lookbackmed_locale
			
			if lookback_mean<lookback_medi and meen>medi: #meen<medi:
				perf = ((seq[packet][seq_len+1][best_locale]/seq[packet][seq_len][best_locale])-1)*100
				performance.append(perf)
				perf1.append(perf)
				#print(perf)
			if lookback_mean>lookback_medi and meen<medi:
				perf = ((seq[packet][seq_len+1][worst_locale]/seq[packet][seq_len][worst_locale])-1)*100
				performance.append(perf)
				perf2.append(perf)
			
			if lookback_mean<lookback_medi and meen<medi: #have tried med, lbbest
				perf = ((seq[packet][seq_len+1][worst_locale]/seq[packet][seq_len][worst_locale])-1)*100
				#perf = -((seq[packet][seq_len+1][best_locale]/seq[packet][seq_len][best_locale])-1)*100
				#perf = 0
				performance.append(perf)
				perf3.append(perf)

			if lookback_mean>lookback_medi and meen>medi:
				perf = ((seq[packet][seq_len+1][med_locale]/seq[packet][seq_len][med_locale])-1)*100
				performance.append(perf)
				perf4.append(perf)
		#  <   >
		#  >   <
		# best, worst, worst, med
		performance = perf1
		print(f"\nLookback Period of {seq_len}:")
		print(f"Total Performance: {sum(performance)}")
		print(f"Compounding Performance: {compounding(perf1,10)}")
		print(f"Scene 1 Perf: {sum(perf1)}  Count: {len(perf1)}")
		print(f"Scene 2 Perf: {sum(perf2)}  Count: {len(perf2)}")
		print(f"Scene 3 Perf: {sum(perf3)}  Count: {len(perf3)}")
		print(f"Scene 4 Perf: {sum(perf4)}  Count: {len(perf4)}")

		# average winner vs average loser
		up_list = []
		down_list = []
		for it in perf1:
			if it>0:
				up_list.append(it)
			else:
				down_list.append(it)
		print("Perf1 Accuracy: ",len(up_list)*100/len(perf1))
		perf1Acc = len(up_list)/len(perf1)

		up_list = []
		down_list = []
		for it in perf2:
			if it>0:
				up_list.append(it)
			else:
				down_list.append(it)
		print("Perf2 Accuracy: ",len(up_list)*100/len(perf2))
		perf2Acc = len(up_list)/len(perf2)

		up_list = []
		down_list = []
		for it in perf3:
			if it>0:
				up_list.append(it)
			else:
				down_list.append(it)
		try:
			print("Perf3 Accuracy: ",len(up_list)*100/len(perf3))
			perf3Acc = len(up_list)/len(perf3)
		except:
			print("Perf3 Accuracy: ",0)
			perf3Acc = 0
		up_list = []
		down_list = []
		for it in perf4:
			if it>0:
				up_list.append(it)
			else:
				down_list.append(it)
		print("Perf4 Accuracy: ",len(up_list)*100/len(perf4))
		perf4Acc = len(up_list)/len(perf4)

		overallAcc.append(perf1Acc*perf2Acc)
		overall_list.append(sum(performance))
		longperf1.append(sum(perf1))
		longperf2.append(sum(perf2))
		longperf3.append(sum(perf3))
		longperf4.append(sum(perf4))
	#print(f"Max Drawdown: {dd}")
	beest = max(overall_list)
	beestplace = overall_list.index(beest) + 10
	print(f"\nOverall Mean Perf: {mean(overall_list)}")
	print(f"Lookback of {beestplace} performed: {beest}")

	print(f"\nPerf1 Mean: {mean(longperf1)}   Median: {median(longperf1)}")
	print(f"Perf2 Mean: {mean(longperf2)}   Median: {median(longperf2)}")
	print(f"Perf3 Mean: {mean(longperf3)}   Median: {median(longperf3)}")
	print(f"Perf4 Mean: {mean(longperf4)}   Median: {median(longperf4)}")

def backtest(seq_len,report=False):

	df = yf.download(stock_list, start=start_date, end=end_date,progress=False)
	df.bfill(axis='rows',inplace=True)
	df.dropna(inplace=True)
	print(df['Adj Close'].tail())
	days = df.index

	print("Collected Tickers")

	sequential_data = []  
	prev_days = deque(maxlen=seq_len+2) 

	for i in df['Adj Close'].values: #df['Adj Close'].values:  # iterate over the values
		prev_days.append([n for n in i]) 
		if len(prev_days) == (seq_len+2):  
			sequential_data.append(np.array(prev_days))
	seq = np.array(sequential_data)

	#print(seq[-1])
	performance = []
	samll_perf = []
	perf1 = []
	perf2 = []
	perf3 = []
	perf4 = []
	for packet in range(seq.shape[0]):
		
		print("\n",days[-(seq.shape[0]-packet)])
		ratio_list = []
		long_list = []
		for item in range(seq.shape[2]):
			ratio = seq[packet][seq_len][item]/seq[packet][seq_len-1][item]
			ratio_list.append(ratio)

		best = max(ratio_list)
		worst = min(ratio_list)
		meen = mean(ratio_list)
		medi = median(ratio_list)
		best_locale = ratio_list.index(best)
		worst_locale = ratio_list.index(worst)
		med_locale = ratio_list.index(medi)

		for item in range(seq.shape[2]):
			ratio = seq[packet][seq_len][item]/seq[packet][0][item]
			long_list.append(ratio)

		lookback_mean = mean(long_list)
		lookback_medi = median(long_list)
		# lookback_best = max(long_list)
		# lookback_worst = min(long_list)
		# # lookbackbest_locale = long_list.index(lookback_best)
		# # lookbackmed_locale = long_list.index(lookback_medi)
		# # lookbackworst_locale = long_list.index(lookback_worst)
		
		if lookback_mean<lookback_medi and meen>medi: #meen<medi:
			perf = ((seq[packet][seq_len+1][best_locale]/seq[packet][seq_len][best_locale])-1)*100
			samll_perf.append(perf/100)
			performance.append(perf)
			perf1.append(perf)
			print("Senario 1")
			print(f"{stock_list[best_locale]} performed: {perf:.2f}")
	
		if lookback_mean>lookback_medi and meen<medi:
			perf = ((seq[packet][seq_len+1][worst_locale]/seq[packet][seq_len][worst_locale])-1)*100
			performance.append(perf)
			samll_perf.append(perf/100)
			perf2.append(perf)
			print("Senario 2")
			print(f"{stock_list[worst_locale]} performed: {perf:.2f}")
		
		if lookback_mean<lookback_medi and meen<medi: #have tried med, lbbest
			#perf = -((seq[packet][seq_len+1][best_locale]/seq[packet][seq_len][best_locale])-1)*100
			perf = ((seq[packet][seq_len+1][worst_locale]/seq[packet][seq_len][worst_locale])-1)*100
			performance.append(perf)
			samll_perf.append(perf/100)
			perf3.append(perf)
			print("Senario 3")
			print(f"{stock_list[best_locale]} performed: {perf:.2f}")

		if lookback_mean>lookback_medi and meen>medi:
			perf = ((seq[packet][seq_len+1][med_locale]/seq[packet][seq_len][med_locale])-1)*100
			#perf=0
			performance.append(perf)
			samll_perf.append(perf/100)
			perf4.append(perf)
			print("Senario 4")
			print(f"{stock_list[med_locale]} performed: {perf:.2f}")
		#  <   >
		#  >   <
	performance = perf1 + perf2# + perf3 + perf4 
	print(f"\nLookback Period of {seq_len}:")
	print(f"Total Performance: {sum(performance)}")
	print(f"Compounding Performance: {compounding(performance,1)}")
	print(f"Scene 1 Perf: {sum(perf1)}  Count: {len(perf1)}")
	print(f"Scene 2 Perf: {sum(perf2)}  Count: {len(perf2)}")
	print(f"Scene 3 Perf: {sum(perf3)}  Count: {len(perf3)}")
	print(f"Scene 4 Perf: {sum(perf4)}  Count: {len(perf4)}")
	
	hit_list = []
	for i in range(len(performance)):
		if performance[i]>=0:
			hit_list.append(1)

	print("\nWin Percentage: ",sum(hit_list)/len(performance))
	print("Best: ",max(performance))
	print("Worst: ",min(performance))
	if report:
		import quantstats as qs 
		ser = pd.Series(samll_perf)
		ser.index = df.index[-len(ser):]
		print(ser)
		qs.reports.html(ser, title="Nova Capital", output="Quantstats_output/returns.html")
	
	else:
		spy = yf.download("SPY",start=start_date,end=end_date,progress=False)
		change_array = np.array(spy['Adj Close'].pct_change()*100)

		plt.plot(range(len(performance)),np.cumsum(np.array(performance)),label='Strat')
		plt.plot(range(len(performance)),np.cumsum(change_array[-len(performance):]),label='Bench')
		plt.legend(loc="upper left")
		plt.show()

def currentPick(sequenceLength):
	# most recent weighting
	seq_len = sequenceLength-1

	df = yf.download(stock_list, start=start_date, end=end_date,progress=False)
	df.bfill(axis='rows',inplace=True)
	df.dropna(inplace=True)
	print(df['Adj Close'].tail())

	print("Collected Tickers")
	
	sequential_data = []  
	prev_days = deque(maxlen=seq_len+2) 

	for i in df['Adj Close'].values: #df['Adj Close'].values:  # iterate over the values
		prev_days.append([n for n in i]) 
		if len(prev_days) == (seq_len+2):  
			sequential_data.append(np.array(prev_days))
	seq = np.array(sequential_data)

	end_of_range = max(range(seq.shape[0]))
	for packet in range(seq.shape[0]):
		
		ratio_list = []
		long_list = []
		for item in range(seq.shape[2]):
			ratio = seq[packet][seq_len+1][item]/seq[packet][seq_len][item]
			ratio_list.append(ratio)

		best = max(ratio_list)
		worst = min(ratio_list)
		meen = mean(ratio_list)
		medi = median(ratio_list)
		best_locale = ratio_list.index(best)
		worst_locale = ratio_list.index(worst)
		med_locale = ratio_list.index(medi)

		for item in range(seq.shape[2]):
			ratio = seq[packet][seq_len+1][item]/seq[packet][0][item]
			long_list.append(ratio)

		lookback_mean = mean(long_list)
		lookback_medi = median(long_list)
		# lookback_best = max(long_list)
		# lookback_worst = min(long_list)
		# lookbackbest_locale = long_list.index(lookback_best)
		# lookbackmed_locale = long_list.index(lookback_medi)
		# lookbackworst_locale = long_list.index(lookback_worst)
		if packet == end_of_range:

			if lookback_mean<lookback_medi and meen>medi: #meen<medi:
				print("Senario 1")
				print(f"Pick: {stock_list[best_locale]}")

			if lookback_mean>lookback_medi and meen<medi:
				print("Senario 2")
				print(f"Pick: {stock_list[worst_locale]}")
			
			if lookback_mean<lookback_medi and meen<medi: #have tried med, lbbest
				print("Senario 3")
				print(f"Long Pick: {stock_list[worst_locale]}")
				print(f"Short Pick: {stock_list[best_locale]}")

			if lookback_mean>lookback_medi and meen>medi:
				print("Senario 4")
				print(f"Pick: {stock_list[med_locale]}")
	#  <   >
	#  >   <

#backtest_loop()
#currentPick(13)
backtest(13,report=False)
benchmark('SPY')

def wfa(start,cycles,leverage):
	#### Walk Forward Analysis ###
	# finds optimal lookback parameter to use in the next cycle
	lookback_used = []
	master_performance = [0]
	for cycle in range(cycles):
		btstart = datetime.strptime(start, "%Y-%m-%d") + timedelta(weeks=cycle*4)
		btend = datetime.strptime(start, "%Y-%m-%d") + timedelta(weeks=52+(cycle*4))
		wfend = datetime.strptime(start, "%Y-%m-%d") + timedelta(weeks=56+(cycle*4))

		whileLoop = True
		while whileLoop:
			df = yf.download(stock_list, start=btstart, end=btend,progress=False)
			whileLoop = df.isnull().values.any()
			if whileLoop:
				print("BT Error")
				time.sleep(30)

		df.bfill(axis='rows',inplace=True)
		df.dropna(inplace=True)
		#print(df.head(15))

		# longperf1 = []
		# longperf2 = []
		overall_list = []
		overallAcc = []

		for seq_len in range(3,21):

			sequential_data = []  
			prev_days = deque(maxlen=seq_len+2) 

			for i in df['Adj Close'].values: #df['Adj Close'].values:  # iterate over the values
				prev_days.append([n for n in i]) 
				if len(prev_days) == (seq_len+2):  
					sequential_data.append(np.array(prev_days))
			seq = np.array(sequential_data)

			performance = []
			perf1 = []
			perf2 = []
			for packet in range(seq.shape[0]):
				
				ratio_list = []
				long_list = []
				for item in range(seq.shape[2]):
					ratio = seq[packet][seq_len][item]/seq[packet][seq_len-1][item]
					ratio_list.append(ratio)

				best = max(ratio_list)
				worst = min(ratio_list)
				meen = mean(ratio_list)
				medi = median(ratio_list)
				best_locale = ratio_list.index(best)
				worst_locale = ratio_list.index(worst)
				med_locale = ratio_list.index(medi)

				for item in range(seq.shape[2]):
					ratio = seq[packet][seq_len][item]/seq[packet][0][item]
					long_list.append(ratio)

				lookback_mean = mean(long_list)
				lookback_medi = median(long_list)
				# lookback_best = max(long_list)
				# lookback_worst = min(long_list)
				# lookbackbest_locale = long_list.index(lookback_best)
				# lookbackmed_locale = long_list.index(lookback_medi)
				# lookbackworst_locale = long_list.index(lookback_worst)
				
				if lookback_mean<lookback_medi and meen>medi: #meen<medi:
					perf = ((seq[packet][seq_len+1][best_locale]/seq[packet][seq_len][best_locale])-1)*100
					performance.append(perf)
					perf1.append(perf)
					#print(perf)
				if lookback_mean>lookback_medi and meen<medi:
					perf = ((seq[packet][seq_len+1][worst_locale]/seq[packet][seq_len][worst_locale])-1)*100
					performance.append(perf)
					perf2.append(perf)
				
			#  <   >
			#  >   <
			# best, worst, worst, med
			performance = perf1# + perf2
			# print(f"\nLookback Period of {seq_len}:")
			# print(f"Total Performance: {sum(performance)}")
			# print(f"Compounding Performance: {compounding(perf1,10)}")
			# print(f"Scene 1 Perf: {sum(perf1)}  Count: {len(perf1)}")
			# print(f"Scene 2 Perf: {sum(perf2)}  Count: {len(perf2)}")

			up_list = []
			down_list = []
			for it in perf1:
				if it>0:
					up_list.append(it)
				else:
					down_list.append(it)
			# print("Perf1 Accuracy: ",len(up_list)*100/len(perf1))
			#perf1Acc = len(up_list)/len(perf1)

			up_list = []
			down_list = []
			for it in perf2:
				if it>0:
					up_list.append(it)
				else:
					down_list.append(it)
			# print("Perf2 Accuracy: ",len(up_list)*100/len(perf2))
			#perf2Acc = len(up_list)/len(perf2)


			overall_list.append(sum(performance))
			#overallAcc.append(perf1Acc*perf2Acc)
			# longperf1.append(sum(perf1))
			# longperf2.append(sum(perf2))

		#print(f"Max Drawdown: {dd}")
		beest = max(overall_list)
		beestplace = overall_list.index(beest) + 3
		lookback_used.append(beestplace)
		# print(f"\nOverall Mean Perf: {mean(overall_list)}")
		# print(f"Lookback of {beestplace} performed: {beest}")

		# print(f"\nPerf1 Mean: {mean(longperf1)}   Median: {median(longperf1)}")
		# print(f"Perf2 Mean: {mean(longperf2)}   Median: {median(longperf2)}")
		btend = btend - timedelta(days=floor(beestplace*1.2858))
		whileLoop = True
		while whileLoop:
			df = yf.download(stock_list, start=btend, end=wfend,progress=False)
			whileLoop = df.isnull().values.any()
			if whileLoop:
				print("WF Error")
				time.sleep(15)

		df.bfill(axis='rows',inplace=True)
		df.dropna(inplace=True)
		# days = df.index
		# print(days[:2])

		seq_len = beestplace
		sequential_data = []  
		prev_days = deque(maxlen=seq_len+2) 

		for i in df['Adj Close'].values: #df['Adj Close'].values:  # iterate over the values
			prev_days.append([n for n in i]) 
			if len(prev_days) == (seq_len+2):  
				sequential_data.append(np.array(prev_days))
		seq = np.array(sequential_data)

		#print(seq[-1])
		performance = []
		#samll_perf = []
		perf1 = []
		perf2 = []

		for packet in range(seq.shape[0]):
			
			#print("\n",days[-(seq.shape[0]-packet)])
			ratio_list = []
			long_list = []
			for item in range(seq.shape[2]):
				ratio = seq[packet][seq_len][item]/seq[packet][seq_len-1][item]
				ratio_list.append(ratio)

			best = max(ratio_list)
			worst = min(ratio_list)
			meen = mean(ratio_list)
			medi = median(ratio_list)
			best_locale = ratio_list.index(best)
			worst_locale = ratio_list.index(worst)
			med_locale = ratio_list.index(medi)

			for item in range(seq.shape[2]):
				ratio = seq[packet][seq_len][item]/seq[packet][0][item]
				long_list.append(ratio)

			lookback_mean = mean(long_list)
			lookback_medi = median(long_list)
			# lookback_best = max(long_list)
			# lookback_worst = min(long_list)
			# # lookbackbest_locale = long_list.index(lookback_best)
			# # lookbackmed_locale = long_list.index(lookback_medi)
			# # lookbackworst_locale = long_list.index(lookback_worst)
			
			if lookback_mean<lookback_medi and meen>medi: #meen<medi:
				perf = ((seq[packet][seq_len+1][best_locale]/seq[packet][seq_len][best_locale])-1)*100
				#samll_perf.append(perf/100)
				performance.append(perf)
				perf1.append(perf)

		
			if lookback_mean>lookback_medi and meen<medi:
				perf = ((seq[packet][seq_len+1][worst_locale]/seq[packet][seq_len][worst_locale])-1)*100
				performance.append(perf)
				#samll_perf.append(perf/100)
				perf2.append(perf)
			

			#  <   >
			#  >   <
		performance = perf1# + perf2
		print(f"\nLookback Period of {beestplace}:")
		print(f"{btend} ===> {wfend}")
		print(f"Total Performance: {sum(performance)}")
		print(f"Compounding Performance: {compounding(performance,leverage)}")
		print(f"Scene 1 Perf: {sum(perf1)}  Count: {len(perf1)}")
		print(f"Scene 2 Perf: {sum(perf2)}  Count: {len(perf2)}")
		up_list = []
		down_list = []
		for it in perf1:
			if it>0:
				up_list.append(it)
			else:
				down_list.append(it)
		try:
			print("Perf1 Accuracy: ",len(up_list)*100/len(perf1))
		except:
			print("Perf1 Accuracy: ",0)

		up_list = []
		down_list = []
		for it in perf2:
			if it>0:
				up_list.append(it)
			else:
				down_list.append(it)
		try:
			print("Perf2 Accuracy: ",len(up_list)*100/len(perf2))
		except:
			print("Perf2 Accuracy: ",0)
		master_performance = master_performance + performance
	print(f"\nOverall Performed: {sum(master_performance)}")
	print(lookback_used)
	print(f"Overall Compounding: {compounding(master_performance,leverage)}")
	plt.plot(range(len(master_performance)),np.cumsum(np.array(master_performance)))
	plt.show()

#wfa(start_date,13,1)