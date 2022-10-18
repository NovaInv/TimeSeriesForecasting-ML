# NovaInv August 10, 2022
# Train and predict using hideen markov model
# Train using returns of index constituents
# Backtest using etf of index

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


plt.rcParams["figure.figsize"] = (12,6)

start_date = '2019-11-12'
end_date = '2022-09-03'
#start_date = '2021-04-01'
#end_date = date.today() + timedelta(days=1)

head_df = pd.read_csv("Datasets/nasdaq100.csv")
stock_list = head_df['Symbol'].tolist()
benchmark = 'QQQ'
useAlgos = True
#stock_list = ['AAPL','MSFT']
#stock_list = ['TQQQ','SQQQ']
#stock_list = ['SMH','SPY','XLB','XLE','XLF','XLI','XLP','XLRE','XLU','XLV','XLY']
#stock_list = ['AAPL','AAL','AMD','BAC','F','META','NIO','NVDA','SPY','TSLA','XLF']
#stock_list = ['AAPL','GOOG','MSFT','AMZN','BRK-B','META','JPM','JNJ','BAC','WMT','V','PFE','HD','DIS','NVDA']
df = yf.download(stock_list, start=start_date, end=end_date,progress=False)['Adj Close']
bench = yf.download(benchmark, start=start_date, end=end_date,progress=False)['Adj Close'].pct_change()
print(df.isnull().values.any())

#print(df[df.isnull().any(axis=1)])
# df.dropna(inplace=True)
df = df/df.shift(1)
df.dropna(inplace=True)
bench.dropna(inplace=True)
perf = []
from hmmlearn import hmm
from sklearn.metrics import accuracy_score 

# train_test_split
X = df[stock_list]#.values.reshape(-1,1)
X_train = X[:int(len(df)*0.75)]
X_test = X[int(len(df)*0.75):]
chg = bench[int(len(df)*0.75):].values # acutal returns in simulated backtest

# hidden markov model from hmm package
model = hmm.GaussianHMM(n_components=2, covariance_type='full', n_iter=1000,random_state=5412)
model.fit(X_train)
y_test = model.predict(X_test)
print(y_test)

# loop through predictions and append performance
for i in range(len(y_test)-1):
	if y_test[i]==1:
		perf.append(chg[i+1]*100)
	else:
		perf.append(-chg[i+1]*100)

# print backtest statistics
print(accuracy_score(np.where(chg>0,1,0),y_test))
print(sum(perf))
print(compounding(perf,3))
# plot performance vs benchmark
plt.plot(range(len(perf)),np.cumsum(np.array(perf)),label='Strat')
plt.plot(range(len(perf)),np.cumsum(np.array(chg*100)[1:]),label='Bench')
plt.legend(loc='upper left')
plt.show()
# plt.figure(figsize=(10, 6))
# plt.scatter(X_test.index, [f'Regime {n+1}' for n in y_test])
# plt.title(f'SPY market regime')
# plt.xlabel("time")
# plt.show()