# NovaInv March 22, 2022
# Use Random Forest Classifer to loop through a list of ticker symbols
# and predict binary change in next day prices

import numpy as np
import pandas as pd
import yfinance as yf 
from datetime import date, timedelta
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from collections import Counter, deque
import time
np.set_printoptions(precision=5)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# list to store and retrieve variables
targ_list=[]
parf_list=[]
proba_list=[]
start_date = '2020-01-01'
end_date = '2022-01-25'
tickers = ['AAPL','AAL','AMC','AMD','BA','BAC','F','FB','IWM','MSFT','NVDA','NIO','QQQ','SPY','TSLA','XLF']

for ticker in tickers:
	lkback = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,26] # list of lookback distances for returns

	feats=[] # store features
	whileLoop = True
	while whileLoop:
		# continue trying to retrieve data until no nan values
		df = yf.download(ticker, start=start_date, end=end_date,progress=False)
		#df.dropna(inplace=True)
		whileLoop = df.isnull().values.any()
		if whileLoop:
			print(f"{ticker} has BT Error")
			time.sleep(30)
	# # vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)
	# # df['vix'] = vix['Adj Close'].pct_change()
	# #df['vixrsi'] = talib.RSI(vix['Adj Close'],timeperiod=7)
	# #print(print(df[df.isna().any(axis=1)]))

	# calculate lookback returns and next day target variables
	for item in lkback:
		df[f"lookback{item}"] = df['Adj Close']/df['Adj Close'].shift(item)-1
		feats.append(f"lookback{item}")
	df['pct_chg'] = df['lookback1'].shift(-1)*100
	df['targ'] = np.where(df['pct_chg'] > 0,1,0) # binary variable for one step ahead change
	df.dropna(inplace=True)
	final = df[-1:]
	df = df[:-1]

	# resample the data and balance between next day increases and declines
	rng = 82
	# df_train = df.sample(int(0.7*len(df)), random_state = rng)
	# df_test = df[~df.index.isin(list(df_train.index))]
	# sample_in = int(min(list(dict(Counter(df_train['targ'])).values()))-1)
	# df_0 = df_train[df_train['targ'] == 0]
	# df_1 = df_train[df_train['targ'] == 1]
	# df_0 =df_0.sample(n=sample_in, random_state = rng)
	# df_1 =df_1.sample(n=sample_in, random_state = rng)
	# df_train = df_0.append(df_1)
	# X_train = df_train[feats]
	# y_train = df_train['targ']
	# X_test =  df_test[feats]
	# y_test = df_test['targ']
	# train_length = int(len(df)*0.7)
	X = df[feats]
	y = df['targ'].values

	# X_train = X[:train_length] 
	# X_test =  X[train_length:]

	# y_train = y[:train_length]
	# y_test = y[train_length:]

	# random forest classifier to predict next day binary change
	clf = RandomForestClassifier(n_estimators=500,max_depth=15,max_features='sqrt',random_state=42,criterion='entropy')
	clf.fit(X,y)
	proba = clf.predict_proba(final[feats])
	print(proba)
	proba_list.append(proba[0][1])
	targ_list.append(final['targ'].values)
	parf_list.append(final['pct_chg'].values)

# store and show prediction and actual results in dataframe
data = {'Ticker':tickers,'Targ': targ_list, 'Predicted':proba_list, 'Perf':parf_list}
ser = pd.DataFrame(data)
ser.sort_values(by=['Predicted'],ascending=False,inplace=True)
print(ser)