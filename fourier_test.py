# NovaInv August 9, 2022
# Fourier Analysis of timeseries and extrapolation used
# to backtest predictions
# Reference: https://ataspinar.com/2020/12/22/time-series-forecasting-with-stochastic-signal-analysis-techniques/

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from collections import Counter
import talib
import math
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def rolling_accuracy(true_,predicted,length):
	# calculate and plot accuracy of predictions over a rolling period
	from collections import deque
	from sklearn.metrics import accuracy_score
	dicc = {"Actual":true_,"Pred":predicted}
	outt = pd.DataFrame(dicc)
	sequential_data = []
	prev_days = deque(maxlen=length)
	prev_dayss = deque(maxlen=length)  

	for i, j in outt.values: 
		prev_days.append(i) 
		prev_dayss.append(j)
		if len(prev_days) == (length):  
			sequential_data.append(accuracy_score(prev_days,prev_dayss))
	plt.bar(range(len(sequential_data)),np.array(sequential_data))
	plt.show()

def compounding(perff,leverage=1):
	# calculate returns compounded
	comp = 1
	for i in range(len(perff)):
		comp = comp*((perff[i]/predictSteps)*leverage/100+1)
	return (comp-1)*100

def backtester(y_test_list,y_pctchg_list,proba_list):
	# backtest using actual and predicted values
	##############  Performance Analysis   ##################
	# list to store and retrieve variables
	perf_list = []
	perfAccurancy = []
	sitout = []
	logic = np.array(y_test_list)
	evalu = np.array(y_pctchg_list)

	# loop through predictions and compare to actual results
	for i in range(len(proba_list)):
		if logic[i] == proba_list[i]:# and proba_list[i] == 1:
			perfAccurancy.append(1)
			if logic[i] == 1:
				perf_list.append(evalu[i])
			else:
				perf_list.append(-evalu[i])
		elif logic[i] != proba_list[i] and proba_list[i] != 2:# and proba_list[i] == 1:
			perfAccurancy.append(0)
			if logic[i] == 1:
				perf_list.append(-evalu[i])
			else:
				perf_list.append(evalu[i])
		else:
			perf_list.append(0)
			sitout.append(1)

	# print performance statistics
	print(f"\nTotal Performance: {sum(perf_list):.2f}     {sum(perf_list)/predictSteps:.2f}")
	print(f"Compounging: {compounding(perf_list,3):.2f}")
	print(f"Performance Accuracy: {sum(perfAccurancy)/len(perfAccurancy)*100:.3f}")
	print(f"Sit Out Days: {sum(sitout)}  {sum(sitout)*100/len(proba_list):.2f}%")
	print(f"Predicted: {Counter(proba_list)}")
	print(f"Actual: {Counter(logic)}")
	print(f"\n{ticker} Performed: {sum(y_pctchg_list):.2f}     {sum(y_pctchg_list)/predictSteps:.2f}")

	# plot performance against benchmark
	graph = pd.DataFrame({"Actual":np.cumsum(np.array(y_pctchg_list)),"Strat":np.cumsum(np.array(perf_list))})
	graph.plot(title=f"Backtest: lengthUsed {lengthUsed}, PredictSteps {predictSteps}",figsize=(6,4))
	plt.show()

def fourierExtrapolation(x, n_predict, harmonicPercent, useTop=True):
	# fast fourier transform of timeseries
	n = x.size
	n_harm = int(n*harmonicPercent)     # number of harmonics in model
	t = np.arange(0, n)

	x_freqdom = np.fft.fft(x)  # detrended x in frequency domain
	f = np.fft.fftfreq(n)      # frequencies
	indexes = list(range(n))

	if useTop:
		# sort indexes by best (highest) amplitudes, lower -> higher
		indexes.sort(key = lambda i: np.max(np.absolute(x_freqdom[i])))
		#print(indexes)
		t = np.arange(0, n + n_predict)
		restored_sig = np.zeros(t.size)
		for i in indexes[-(1 + n_harm * 2):]:
		    ampli = np.absolute(x_freqdom[i]) / n   # amplitude
		    phase = np.angle(x_freqdom[i])          # phase
		    restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
		return restored_sig 
	else:
		# sort indexes by frequency, lower -> higher
		indexes.sort(key = lambda i: np.absolute(f[i]))
		#print(indexes)
		t = np.arange(0, n + n_predict)
		restored_sig = np.zeros(t.size)
		for i in indexes[:(1 + n_harm * 2)]:
			# loop through and reconstruct signal
		    ampli = np.absolute(x_freqdom[i]) / n   # amplitude
		    phase = np.angle(x_freqdom[i])          # phase
		    restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
		return restored_sig 

def premier(series,stochlen,smoothlen):
	# premier stochastic oscillator
	hi = series.rolling(stochlen).max()
	lo = series.rolling(stochlen).min()
	sk = (series - lo)*100 / (hi-lo) # standardize
	len_ = round(math.sqrt(smoothlen))
	nsk = 0.1*(sk-50)
	ss = talib.EMA(talib.EMA(nsk,len_),len_) # smoothing
	expss = np.exp(ss)
	pro = ((expss - 1) / (expss + 1) + 1) / 2 # oscillating
	return pro

lookback = 5
predictSteps = 1
lengthUsed = 500
ticker = 'SPY'

# get data from yahoo fiannce
df = yf.download(ticker,start='2016-01-01',end='2022-09-03',progress=False)
df['logclose'] = np.log(df['Adj Close'])
df['logdiff'] = np.log(df['Adj Close']/df['Adj Close'].shift(predictSteps)) # log returns of specific period
df['fracdiff'] = (df['Adj Close'].shift(1) * 0.5 + df['Adj Close'].diff() * 0.5) # simple fractional differencing

df['targ'] = np.where(df['logdiff'] > 0, 1, 0) # binary target
df['std'] = df['logdiff'].rolling(window=20).std()

df['pct_chg'] = ((df['Adj Close']/df['Adj Close'].shift(5) - 1) * 100)#.shift(predictSteps)
df['pct_chg2'] = ((df['Adj Close']/df['Adj Close'].shift(4) - 1) * 100)
df['B'] = df['pct_chg'] - df['pct_chg2'].shift(1)
df['pct_chg'] = ((df['Adj Close']/df['Adj Close'].shift(1) - 1) * 100)
df['a'] = np.where(df['pct_chg'] > 0 , 1, 0)

# df['ma'] = df['Adj Close'].rolling(window=lookback).mean()
df['logma'] = df['logclose'].rolling(window=lookback).mean()
df['minus'] = df['logclose'] - df['logma'] # difference between log price and moving average of log prices
df['b'] = np.where(df['minus'] > 0, 1, 0)
# df['regdiff'] = df['Adj Close'] - df['ma']
# df['loggeddiff'] = df['logclose'] - df['logma']
df['rsi'] = talib.RSI(df['Adj Close'],timeperiod=5)/100
df['pro'] = premier(df['rsi'],8,25)
df.dropna(inplace=True)
#print(df[['logdiff','std']].tail(50))


###### Fourier Analysis ########
# fft_y_ = np.fft.fft(np.array(df['minus']))
# fft_y = np.abs(fft_y_[:len(fft_y_)//2])

# fft_x_ = np.fft.fftfreq(len(df['minus']))
# fft_x = fft_x_[:len(fft_x_)//2]

###Plot frequency spectrum
# fig, ax = plt.subplots(figsize=(8,3))
# ax.plot(fft_x, fft_y)
# ax.set_ylabel('Amplitude', fontsize=14)
# ax.set_xlabel('Frequency [1/day]', fontsize=14)
# plt.title(f"{ticker}")
# plt.show()
# 
#pred = fourierExtrapolation(x=np.array(df['minus']),n_predict=predictSteps, harmonicPercent=0.15,useTop=True)
# stop
# length1 = int(len(df)*0.15)
# fft_y_copy = fft_y_.copy()

# fft_y_copy[length1:-length1] = 0
# inverse_fft = np.fft.ifft(fft_y_copy,n=len(df)+predictSteps)
# print(len(inverse_fft))
# print(len(pred))
# plt.plot(range(50),inverse_fft[-50:],label="ifft")
# plt.plot(range(50),pred[-50:],label='func')
# plt.plot(range(50),np.array(df['minus'])[-50:],label='Acutal')
# plt.legend(loc="upper left")
# plt.show()
# stop


# stop
from sklearn.preprocessing import MinMaxScaler, StandardScaler
pred_list = [0]
binary_pred_list = [0]
actual_list = [0]
binary_actual_list = [0]
pctchg_list = [0]

# loop through time steps and predict ahead by 'predictSteps'

for i in range(0,len(df)-lengthUsed+1,predictSteps):
	#scaler = StandardScaler()
	run = df[i:i+lengthUsed].copy(deep=True) # window to use
	#run[['logdiff','logma']] = scaler.fit_transform(run[['logdiff','logma']])
	out = np.array(run['logdiff'].iloc[-predictSteps:]) # test array
	chg = np.array(run['pct_chg'].iloc[-predictSteps:]) # test pct chg array
	inn = run[:-predictSteps] # train array


	pred = fourierExtrapolation(x=np.array(inn['logdiff']),n_predict=predictSteps, harmonicPercent=0.15, useTop=True)
	# print(pred[-predictSteps:])
	# print(out['logdiff'].tail(predictSteps))
	for p in range(predictSteps):

		pctchg_list.append(chg[-(predictSteps-p)])
		# recal = pred[-1] - inn['pct_chg2'].iloc[-1]
		# if recal > 0:
		# 	binary_pred_list.append(1)
		# else:
		# 	binary_pred_list.append(0)

		# if i == len(df)-lengthUsed:
		# 	print(df.index[i+lengthUsed-1])
		# 	print(out[-1])
		# 	print(pred[-1])
		# 	print(inn['pct_chg2'].iloc[-1])
		# 	print(chg[-1])

		if pred[-(predictSteps-p)] > pred_list[-1]:#pred[-(predictSteps-p-1)]:
			binary_pred_list.append(1)
		else:
			binary_pred_list.append(0)

		if chg[-(predictSteps-p)] > 0:
			binary_actual_list.append(1)
		else:
			binary_actual_list.append(0)

		pred_list.append(pred[-(predictSteps-p)])
		actual_list.append(out[-(predictSteps-p)])


# print results of backtest in dataframe
results = pd.DataFrame({"Pred":pred_list,"Actual":actual_list,"BinaryPred":binary_pred_list,"BinaryActual":binary_actual_list})
print(results.tail(20))

# plot results of backtest
# #rolling_accuracy(binary_actual_list,binary_pred_list,50)
backtester(binary_actual_list,pctchg_list,binary_pred_list)
plt.plot(range(100),pred_list[-100:],label='Pred')
plt.plot(range(100),actual_list[-100:],label='Actual')
plt.legend(loc="upper left")
plt.show()
# # plt.plot(range(50),pred[-50:],label='Pred')
# plt.plot(range(50),np.array(df['logdiff'].iloc[-50:]),label='Actual')
# plt.legend(loc="upper left")
# plt.show()
# print(pred[-1])
# print(out['logdiff'])
# print(len(pred))
# print(len(df['logdiff']))



## Reconstruct and plot from portion of frequency transform
# length = len(fft_x)
# length1 = int(length*0.2)
# length2a = int(length*0.5)
# length2b = int(length*0.7)
# length3 = int(length*0.84)

	
# fig, axarr = plt.subplots(figsize=(10,7), nrows=2)
# axarr[0].plot(fft_x, fft_y, linewidth=2, label='full spectrum')
# axarr[0].plot(fft_x[:length1], fft_y[:length1], label='first peak')
# axarr[0].plot(fft_x[length2a:length2b], fft_y[length2a:length2b], label='second peak')
# axarr[0].plot(fft_x[length3:], fft_y[length3:], label='remaining peaks')
# axarr[0].legend(loc='upper left')
 
# filter frequencies
# fft_y_copy1 = fft_y_.copy()
# fft_y_copy2 = fft_y_.copy()
# fft_y_copy3 = fft_y_.copy()
# fft_y_copy1[length1:-length1] = 0
# fft_y_copy2[:length2a] = 0
# fft_y_copy2[length2b:-length2b] = 0
# fft_y_copy2[-length2a:] = 0
# fft_y_copy3[:length3] = 0
# fft_y_copy3[-length3:] = 0
# # fft_y_copy2[:length3a] = 0
# # fft_y_copy2[length3b:-length3b] = 0
# # fft_y_copy2[-length3a:] = 0

# reconstructed filtered signal
# inverse_fft = np.fft.ifft(fft_y_)
# inverse_fft1 = np.fft.ifft(fft_y_copy1)
# inverse_fft2 = np.fft.ifft(fft_y_copy2)
# inverse_fft3 = np.fft.ifft(fft_y_copy3)

# plot reconstructed signal
# axarr[1].plot(inverse_fft, label='inverse of full spectrum')
# axarr[1].plot(inverse_fft1, label='inverse of 1st peak')
# axarr[1].plot(inverse_fft2, label='inverse of 2nd peak')
# axarr[1].plot(inverse_fft3, label='inverse of remaining peaks')
# axarr[1].legend(loc='upper left')
# plt.show()

