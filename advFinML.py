# NovaInv January 30, 2022
# Use fractional differencing to construct features to be
# used in modeling and machine learning
# References: https://github.com/BlackArbsCEO/Adv_Fin_ML_Exercises/blob/master/notebooks/05.%20Fractionally%20Differentiated%20Features.ipynb
#             Advances in Financial Machine Learning - Marcos Lopez de Prado

import numpy as np
import pandas as pd
import yfinance as yf 
from datetime import date, timedelta
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from collections import Counter, deque
# import pmdarima as pm
# from statsmodels.tsa.arima_model import ARIMA 
# from statsmodels.tsa.stattools import adfuller
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#import talib
import time
import random
np.set_printoptions(precision=5)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def compounding(perff,leverage=1):
	# calculate compounding return
	comp = 1
	for i in range(len(perff)):
		comp = comp*(perff[i]*leverage/100+1)
	return (comp-1)*100

def mostRecent(ticker,adjustment=1):
	# retrieve most recent time series of ticker
	startDate = date.today() - timedelta(days=380)
	endDate = date.today() + timedelta(days=adjustment)
	whileLoop = True
	while whileLoop: # continue trying to retrieve data until no nan values
		recent = yf.download(ticker,start=startDate,end=endDate,progress=False)
		whileLoop = recent.isnull().values.any()

	lkback = [1,2,3,4,5,6,7,8,9,10,11,12,20,120,140,160,240,252] # lookback list used to generate returns
	feats = []
	for item in lkback:
		recent[f"lookback{item}"] = recent['Adj Close']/recent['Adj Close'].shift(item)-1
		#recent[f"lookback{item}"] = ((recent['Open']-recent['Close'].shift(item))/recent['Close'].shift(item))
		feats.append(f"lookback{item}")

	recent.dropna(inplace=True)
	print("\n",recent.index[-1])
	return np.array(recent[feats].iloc[-1:])


def getWeights_FFD(d,thres):
	# generate weights for fractional differencing given threshold
    w,k=[1.],1
    while True:
        w_=-w[-1]/k*(d-k+1)
        if abs(w_)<thres:break
        w.append(w_);k+=1
    return np.array(w[::-1]).reshape(-1,1)

#-----------------------------------------------------

def fracDiff_FFD(series,d,thres=1e-5):
	# calculate fraction differencing for given time series
    # Constant width window (new solution)
    w = getWeights_FFD(d,thres)
    width = len(w)-1
    df={}
    for name in series.columns:
        seriesF, df_=series[[name]].fillna(method='ffill').dropna(), pd.Series()
        for iloc1 in range(width,seriesF.shape[0]):
            loc0,loc1=seriesF.index[iloc1-width], seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1,name]): continue # exclude NAs
            #print(f'd: {d}, iloc1:{iloc1} shapes: w:{w.T.shape}, series: {seriesF.loc[loc0:loc1].notnull().shape}')
            df_.loc[loc1]=np.dot(w.T, seriesF.loc[loc0:loc1])[0,0]
        df[name]=df_.copy(deep=True)
        #print(df)
    df=pd.concat(df,axis=1)
    return df 

def get_optimal_ffd(x,ds, t=1e-5):
	# get the optimal fraction differencing cutoff using various metrics
    
    cols = ['adfStat','pVal','lags','nObs','95 conf']#,'corr']
    out = pd.DataFrame(columns=cols)
    
    for d in ds:
        try:
            #dfx = fracDiff(x.to_frame(),d,thres=1e-5)
            dfx = fracDiff_FFD(x,d,thres=t)
            dfx = adfuller(dfx, maxlag=1,regression='c',autolag=None)
            out.loc[d]=list(dfx[:4])+[dfx[4]['5%']]
        except Exception as e:
            print(f'{d} error: {e}')
            break
    return out


# # end_date = date.today() + timedelta(days=1)
# sector_list = ['DIA','QQQ','SPY','XLF','XLP','XLV']

# df = yf.download('SPY',start=start_date, end=end_date, progress=False)
# df.drop(['High','Low','Volume'],axis=1,inplace=True)
# vix = yf.download('XLE',start=start_date,end=end_date,progress=False)
# vix.drop(['High','Low','Volume'],axis=1,inplace=True)
# vix['pct_chg'] = vix['Adj Close'].pct_change()*100
# vix.dropna(inplace=True)
# for item in sector_list:
# 	df[item] = yf.download(item,start=start_date, end=end_date, progress=False)['Adj Close']
# 	df[item] = np.log(df[item]/df[item].shift(1))
# 	#data[item] = np.where(data[item] > 0, 1, 0)
# 	time.sleep(7)
# 	print(item)

# df['Futreturns'] = df['Adj Close'].pct_change().shift(-1) #close to close change
# df['Futreturns'] = ((df['Open']-df['Close'].shift(1))/df['Close'].shift(1))*100
# df['Futret'] = np.where(df['Futreturns'] > 0, 1, 0)
# df['pct_chg'] = df['Adj Close'].pct_change()*100
# df['pct_logic'] = np.where(df['pct_chg'] > 0, 1, 0)
# # df.dropna(inplace=True)
# df['logged'] = np.log(df['Adj Close'])
# df['openPct_chg'] = df['Open'].pct_change()*100
# df.drop(['Open','Close','Adj Close','High','Low','Volume'],axis=1,inplace=True)
# print(df.head())

# Sectoin for ARIMA model
########################################################
useExog = False
seq_len = 90
#df = pd.read_csv("Datasets/SmallSheet_cont_spy_gap.csv")
d_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
#print(get_optimal_ffd(df[['logged']],d_list,1e-3))

# df['new'] = fracDiff_FFD(df[['Adj Close']],d=0.4, thres=1e-3)
# df.dropna(inplace=True)
# df['temp'] = df['new'].pct_change()*100
# df['temp2'] = np.where(df['temp'] > 0,1,0)
#df.dropna(inplace=True)
#print(df.tail(10))
#mask = df['pct_logic'] != df['temp2']
#print(df[mask])
#print(accuracy_score(df['pct_logic'],df['temp2'])*100)


# plot_acf(df['new'],lags=30)
# plot_pacf(df['new'],lags=30)
# plt.show()
# stop
# model = pm.auto_arima(df[-180:]['new'], start_p=1, start_q=1,
#                       test='adf',       # use adftest to find optimal 'd'
#                       max_p=3, max_q=3, # maximum p and q
#                       m=1,              # frequency of series
#                       d=None,           # let model determine 'd'
#                       seasonal=False,   # No Seasonality
#                       start_P=0, 
#                       D=0, 
#                       trace=True,
#                       error_action='ignore',  
#                       suppress_warnings=True, 
#                       stepwise=True)

# print(model.summary())
# stop
# plot_acf(np.array(df['Adj Close'].values),lags=30)
# plot_pacf(df['Adj Close'].values,lags=30)
# plt.show()

#plt.plot(df['new'])
#plt.plot(df['Adj Close'])
#plt.show()
#seqLenList = [10,15,20,25,30,35,40,50,60]
# sector_list = ['XLB','XLE','XLF','XLI','XLK','XLP','XLRE','XLU','XLV','XLY']
# sector_list = ['XLE','XLF','XLK','XLP','XLY']
#seqLenList = [1,4,5,6,7,8,9,10,11,12,13]

# df = pd.read_csv('Datasets/SectorSheet_bestTarg.csv')
# this = df['targ'].shift(-1)
# df['targ'] = this
# df.dropna(inplace=True)
#for seq_len in seqLenList:
# sequential_data = []
# perf_list = []
# prev_days = deque(maxlen=seq_len) 

# for i in this[:-1].values:  # iterate over the values
# 	prev_days.append([n for n in i]) 
# 	if len(prev_days) == (seq_len):  
# 		sequential_data.append(np.array(prev_days))
# seq = np.array(sequential_data)

# if useExog:
# 	exsequential_data = []
# 	exprev_days = deque(maxlen=seq_len)
# 	for i in df[:-1][['Open','Close']].values:  # iterate over the values
# 		exprev_days.append([n for n in i])
# 		if len(exprev_days) == (seq_len): 
# 			exsequential_data.append(np.array(exprev_days))

# 	exseq = np.array(exsequential_data)
# 	#print(len(seq),len(df[seq_len:]))
# really = np.array(df[-len(seq):]['Futret'].values)
# 	#pred_list = []

# result =[]
# fc_ = []
# model = pm.ARIMA(order=(3,0,0),suppress_warnings=True)
# for i in range(len(df)-seq_len):
# 	if useExog:
# 		n_periods = 2
# 		model.fit(seq[i], exogenous=exseq[i])
# 		fc1, conf = model.predict(n_periods=n_periods,exogenous=exseq[i][-2:], return_conf_int=True)
# 		#print(fc)
# 		#print("\n")
# 		#print(np.mean(seq[i]), fc1[0], conf[0][1]-conf[0][0])
# 		#pred_list.append(fc[0])
# 	else:
# 		n_periods = 1
# 		model.fit(seq[i])
# 		fc2, conf = model.predict(n_periods=n_periods, return_conf_int=True)
# 		#print(np.mean(seq[i]), fc2, conf[0][1]-conf[0][0])
# 		fc_.append(fc2[0])
# 		if fc2[0]>seq[i][-1]:
# 			result.append(1)
# 			#print("Model Predicted: ",1)
# 		else:
# 			result.append(0)
# 			#print("Model Predicted: ",0)
# 		#print("Actual: ",really[i])
# 		#print("\n")
		
# 		#pred_list.append(fc)
# result = np.array(result)
# print(result)
# print("Seq Len: ",seq_len)
# print("Accuracy Score: ", accuracy_score(df[-len(result):]['pct_logic'].values,result)*100)
# plt.plot(np.array(fc_[-100:]))
# plt.plot(np.array(df[-100:]['new']))
# plt.show()
# evalu = np.array(df[-len(result):]['Futreturns']*100)
# logic = np.array(df[-len(result):]['Futret'])
# for i in range(len(result)):
# 	if logic[i] == result[i]:
# 		if logic[i] == 1:
# 			perf_list.append(evalu[i])
# 		else:
# 			perf_list.append(-evalu[i])
# 	else:
# 		if logic[i] == 1:
# 			perf_list.append(-evalu[i])
# 		else:
# 			perf_list.append(evalu[i])

# print("Total Performance: ", sum(perf_list))
# plt.bar(range(len(perf_list)),np.cumsum(np.array(perf_list)))
# plt.show()
########################################################

# df = pd.read_csv("Datasets/SectorSheet_cont_spy.csv")
# #print(df.head())
# df.rename(columns={'Futreturns':'pct_chg','Futret':'targ'},inplace=True)
# #df['targ'] = df['targ'].shift(-1)
# df.dropna(inplace=True)
# #df = df.astype({"targ":'int64'})
# feats = ['SMH','XLB','XLE','XLF','XLI','XLK','XLP','XLRE','XLU','XLV','XLY']

##############  Data Collection and Feature Generation ##############
start_date = '2015-01-01'
end_date = '2022-02-19'
#lkback = [1,2,3,4,5,6,7,8,9,10,11,12,20,120,140,160,240,252]
lkback = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,26]
# feats = ['rsi','lr5','stdev','adx']#,'mom']
feats=[]
ticker = 'AAPL'
df = yf.download(ticker,start=start_date,end=end_date,progress=False,interval='1wk')
print(df.isnull().values.any())
# # vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)
# # df['vix'] = vix['Adj Close'].pct_change()
# #df['vixrsi'] = talib.RSI(vix['Adj Close'],timeperiod=7)
# #print(print(df[df.isna().any(axis=1)]))


for item in lkback:
	df[f"lookback{item}"] = df['Adj Close']/df['Adj Close'].shift(item)-1
	feats.append(f"lookback{item}")
df['pct_chg'] = df['lookback1'].shift(-1)*100
df['targ'] = np.where(df['pct_chg'] > 0,1,0)
# df['rsi'] = talib.RSI(df['Adj Close'],timeperiod=14)
# # df['rsi7'] = talib.RSI(df['Adj Close'],timeperiod=7)
# # df['rsi4'] = talib.RSI(df['Adj Close'],timeperiod=4)
# # df['rsi28'] = talib.RSI(df['Adj Close'],timeperiod=28)
# #df['macd'], _, _ =talib.MACD(df['Adj Close'],fastperiod=12,slowperiod=26,signalperiod=9)
# df['lr5'] = talib.LINEARREG_SLOPE(df['Adj Close'], timeperiod=5)
# df['stdev'] = talib.STDDEV(df['Adj Close'],timeperiod=20)
# df['adx'] = talib.ADX(df['High'], df['Low'] ,df['Close'],timeperiod=14)

#df['mom'] = talib.MOM(df['Adj Close'], timeperiod=10)
#print(df.tail(10))


# for item in lkback:
# 	df[f"lookback{item}"] = ((df['Open']-df['Close'].shift(item))/df['Close'].shift(item))
# 	feats.append(f"lookback{item}")
# df['pct_chg'] = df['lookback1'].shift(-1)*100
# df['targ'] = np.where(df['pct_chg'] > 0, 1, 0)
df.dropna(inplace=True)
# Use Sci-Kit Learn algorithims
##############  Model Packages  ##############
from sklearn.linear_model import RidgeClassifier, LogisticRegression, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.naive_bayes import GaussianNB

##############  Split up data  ##############
train_length = int(len(df)*0.7)
X = df[feats]
y = df['targ'].values
ee = df['pct_chg'].values
ee = ee[train_length:]

X_train = X[:train_length] 
X_test =  X[train_length:]

y_train = y[:train_length]
y_test = y[train_length:]
#spyPerf = np.array(df[train_length:]['pct_chg'])
rng = 82
df_train = df.sample(int(0.7*len(df)), random_state = rng)
df_test = df[~df.index.isin(list(df_train.index))]
# #df_train = x[:train_length]

##############  Resample data and balance  ##############
sample_in = int(min(list(dict(Counter(df_train['targ'])).values()))-1)
df_0 = df_train[df_train['targ'] == 0]
df_1 = df_train[df_train['targ'] == 1]
# # df_2 = df_train[df_train['targ'] == 2]
# # df_3 = df_train[df_train['targ'] == 3]
# # df_4 = df_train[df_train['targ'] == 4]
df_0 =df_0.sample(n=sample_in, random_state = rng)
df_1 =df_1.sample(n=sample_in, random_state = rng)
# # df_2 =df_2.sample(n=sample_in, random_state = rng)
# # df_3 =df_3.sample(n=sample_in, random_state = rng)
# # df_4 =df_4.sample(n=sample_in, random_state = rng)
df_train = df_0.append(df_1)
# # df_train = df_train.append(df_2)
# # df_train = df_train.append(df_3)
# # df_train = df_train.append(df_4)
# df_train = df_train.sample(frac=1, random_state=rng)

X_train = df_train[feats]
y_train = df_train['targ']
X_test =  df_test[feats]
y_test = df_test['targ']
print(Counter(y_train))
##############  Model Training and Testing  ##############
# using differenc classifiers
#clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=20),n_estimators=1000,random_state=42)
kclf = LogisticRegression()
hclf = GaussianNB() 
bclf = RandomForestClassifier(n_estimators=1250,max_depth=20,max_features='sqrt',random_state=42)
clf = RandomForestClassifier(n_estimators=500,max_depth=15,max_features='sqrt',random_state=42,criterion='entropy')
#bclf = VotingClassifier(estimators=[('gnb',gclf),('rf',rclf)],voting='soft')

clf.fit(X_train,y_train)
pred = clf.predict(X_test)
print("Test values: ",Counter(y_test))
print("Predicted values: ",Counter(pred))
print("Accuracy: ",accuracy_score(y_test,pred)*100)
print(pred[-20:])
print(y_test[-20:])
probas = clf.predict_proba(X_test) #predict probabilities of 0,1
print(probas[-20:])

y_test = np.array(y_test)
# different thresholds to use
tobaThreshlist = [0.5,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.6,0.61,0.62,0.63,0.64,0.65,0.66,0.67]
for tobaThresh in tobaThreshlist:
	# loop through each threshold to see the performance
	toba = []
	newy = []
	newEvalu = []
	for t in range(len(probas)):
		# loop through each prediction to see if it is above or below threshold
		if probas[t][1] > tobaThresh:
			toba.append(0)
			newy.append(y_test[t])
			newEvalu.append(ee[t])
		elif probas[t][1] < (1 - tobaThresh):
			toba.append(1)
			newy.append(y_test[t])
			newEvalu.append(ee[t])
		else:
			z=1
	print("\nThreshold: ",tobaThresh)
	print("Percent in: ", len(toba)*100/len(y_test))
	print("Accuracy: ",accuracy_score(newy,toba)*100)

	perf_list = []
	perfAccurancy = []
	sitout = []
	logic = newy
	evalu = newEvalu*100

	for i in range(len(toba)):
		# loop through predictoins again and append the performances
		if logic[i] == toba[i]:
			perfAccurancy.append(1)
			if logic[i] == 1:
				perf_list.append(evalu[i])
			else:
				perf_list.append(-evalu[i])
		elif logic[i] != toba[i] and toba[i] != 2:
			perfAccurancy.append(0)
			if logic[i] == 1:
				perf_list.append(-evalu[i])
			else:
				perf_list.append(evalu[i])
		else:
			perf_list.append(0)
			sitout.append(1)

	# calculate and print performance statistics
	print("\nTotal Performance: ", sum(perf_list))
	print("Compounging: ", compounding(perf_list,10))
	print("Proba Thresh: ", tobaThresh)
	print("Performance Accuracy: ", sum(perfAccurancy)/len(perfAccurancy)*100)
	print(f"Sit Out Days: {sum(sitout)}  {sum(sitout)*100/len(toba)}%")

##############  Plot Feature Importances  ##############
stop
featureArray = np.array(clf.feature_importances_)
plt.barh(feats,featureArray)
plt.xlim(np.min(featureArray),np.max(featureArray))
plt.show()

##############  Make Trinary Results Logic  ##############
probaThresholdList = [0.5,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.6]
#probaThresholdList = [0.5,0.53,0.56,0.59,0.62,0.65,0.68,0.7]
for probaThreshold in probaThresholdList:
	#probaThreshold = 0.53
	proba_list = []
	for i in range(len(probas)):
		if probas[i][1] > probaThreshold:
			proba_list.append(1) # go long
		elif probas[i][1] < (1 - probaThreshold):
			proba_list.append(0) # go short
		else:
			proba_list.append(2) # sitout

	##############  Performance Analysis   ##################
	perf_list = []
	perfAccurancy = []
	sitout = []
	logic = np.array(df[-len(pred):]['targ'])
	evalu = np.array(df[-len(pred):]['pct_chg'])*100

	for i in range(len(proba_list)):
		if logic[i] == proba_list[i]:
			perfAccurancy.append(1)
			if logic[i] == 1:
				perf_list.append(evalu[i])
			else:
				perf_list.append(-evalu[i])
		elif logic[i] != proba_list[i] and proba_list[i] != 2:
			perfAccurancy.append(0)
			if logic[i] == 1:
				perf_list.append(-evalu[i])
			else:
				perf_list.append(evalu[i])
		else:
			perf_list.append(0)
			sitout.append(1)

	print("\nTotal Performance: ", sum(perf_list))
	print("Compounging: ", compounding(perf_list,1))
	print("Proba Thresh: ", probaThreshold)
	print("Performance Accuracy: ", sum(perfAccurancy)/len(perfAccurancy)*100)
	print(f"Sit Out Days: {sum(sitout)}  {sum(sitout)*100/len(proba_list)}%") # what percentage of time is strategy out of the market
#print(f"\nSPY Performed: {np.sum(spyPerf)}")
up_list = []
down_list = []
for it in perf_list:
	if it>0:
		up_list.append(it)
	if it<0:
		down_list.append(it)

# average win and loss of strategy
print("\nAverage up: ",np.mean(np.array(up_list)))
print("Average down: ", np.mean(np.array(down_list)))

# plt.plot(range(len(perf_list)),np.cumsum(np.array(perf_list)))
# plt.show()
# print new dataframe showing predictions to actuality
data = {'Targ': df[-len(pred):]['targ'].values, 'Predicted':proba_list, 'Perf':np.cumsum(np.array(perf_list))}
ser = pd.DataFrame(data)
ser.index = df.index[-len(ser):]
print(ser.tail(20))
