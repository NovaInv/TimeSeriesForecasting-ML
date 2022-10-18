# NovaInv September 7, 2022
# Convolutional Auto Encoder Anomaly detection
# take (x,x) array of one type and train autoencoder
# use reconstruction error in test to determine one type or the other
# Reference: https://medium.com/@judewells/image-anomaly-detection-novelty-detection-using-convolutional-auto-encoders-in-keras-1c31321c10f2

import yfinance as yf
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import math
import time
from collections import Counter
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import talib
pd.set_option('display.max_rows', None)

def conf_matrix(true,pred):
	# confusion matrix
	import seaborn as sns
	from sklearn.metrics import confusion_matrix
	cf_matrix = confusion_matrix(true,pred)
	group_names = ['True Neg','False Pos','False Neg','True Pos']

	group_counts = ["{0:0.0f}".format(value) for value in
	                cf_matrix.flatten()]

	group_percentages = ["{0:.2%}".format(value) for value in
	                     cf_matrix.flatten()/np.sum(cf_matrix)]

	labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
	          zip(group_names,group_counts,group_percentages)]

	labels = np.asarray(labels).reshape(2,2)

	ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')

	ax.set_title('Seaborn Confusion Matrix with labels\n\n');
	ax.set_xlabel('\nPredicted Values')
	ax.set_ylabel('Actual Values ');

	## Ticket labels - List must be in alphabetical order
	ax.xaxis.set_ticklabels(['False','True'])
	ax.yaxis.set_ticklabels(['False','True'])

	## Display the visualization of the Confusion Matrix.
	plt.show()

def compounding(perff,leverage=1):
	# calculate returns compounded
	comp = 1
	for i in range(len(perff)):
		comp = comp*((perff[i]/periodToPredict)*leverage/100+1)
	return (comp-1)*100

class AutoEncoder(Model):
	def __init__(self):
		super(AutoEncoder, self).__init__()
		self.encoder = tf.keras.Sequential([
			layers.Input(shape=(20, 20,1)),
			layers.Conv2D(14, (3, 3), kernel_initializer='lecun_normal',activation='selu', padding='same', strides=2),
			layers.Conv2D(7, (3, 3), kernel_initializer='lecun_normal',activation='selu', padding='same', strides=2)])

		self.decoder = tf.keras.Sequential([
			layers.Conv2DTranspose(7, kernel_size=3, strides=2, kernel_initializer='lecun_normal',activation='selu', padding='same'),
			layers.Conv2DTranspose(14, kernel_size=3, strides=2, kernel_initializer='lecun_normal',activation='selu', padding='same'),
			layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])

	def call(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded


start_date = '2019-04-01'
end_date = '2022-09-08'
#stockList = ['SMH','SPY','XLB','XLE','XLF','XLI','XLP','XLRE','XLU','XLV','XLY','QQQ','V','WMT']
stockList = ['SPY','QQQ','HYG','IWM','SLV','AAPL','TSLA','AMD','NVDA','AMZN','BAC','GOOGL','SNAP','BABA','META']
periodToPredict = 10 # how far ahead to predict

def getToday(stockList):
	# get most recent data to test
	# not used currently
	from datetime import date, timedelta
	end_date = date.today() + timedelta(days=1)
	start_date = end_date - timedelta(days=40)
	
	scaler = MinMaxScaler()
	use = ['sma','ema','dema','adx','aroondown','aroonup','cci','cmo','macd','mfi','mom','roc','rsi','stoch', 'stochd','trix','obv','atr','wpr','cad']
	X = []
	y = []
	for ticker in stockList:
		print(f'Getting {ticker} data.')
		time.sleep(3)
		df = yf.download(ticker,start=start_date,end=end_date,progress=False)
		print(df.isnull().values.any())

		df['sma'] = talib.SMA(df['Adj Close'], timeperiod=20)
		df['ema'] = talib.EMA(df['Adj Close'], timeperiod=20)
		df['dema'] = talib.DEMA(df['Adj Close'], timeperiod=20)
		df['adx'] = talib.ADX(df['High'],df['Low'],df['Close'], timeperiod=14)
		df['aroondown'], df['aroonup'] = talib.AROON(df['High'],df['Low'],timeperiod=14)
		df['cci'] = talib.CCI(df['High'],df['Low'],df['Close'], timeperiod=14)
		df['cmo'] = talib.CMO(df['Adj Close'], timeperiod=14)
		df['macd'], _, _ = talib.MACDFIX(df['Adj Close'])
		df['mfi'] = talib.MFI(df['High'],df['Low'],df['Close'],df['Volume'], timeperiod=14)
		df['mom'] = talib.MOM(df['Adj Close'], timeperiod=10)
		df['roc'] = talib.ROC(df['Adj Close'], timeperiod=10)
		df['rsi'] = talib.RSI(df['Adj Close'], timeperiod=14)
		df['stoch'], df['stochd'] = talib.STOCHF(df['High'],df['Low'],df['Close'])
		df['trix'] = talib.TRIX(df['Adj Close'], timeperiod=30)
		df['obv'] = talib.OBV(df['Adj Close'],df['Volume'])
		df['atr'] = talib.ATR(df['High'],df['Low'],df['Close'], timeperiod=14)
		df['wpr'] = talib.WILLR(df['High'],df['Low'],df['Close'], timeperiod=14)
		df['cad'] = talib.ADOSC(df['High'],df['Low'],df['Close'],df['Volume'])

		df.dropna(inplace=True)

		seq_len = len(use)
		prev_days = deque(maxlen=seq_len)

		for i in df[use].values:
			prev_days.append([n for n in i])
			if len(prev_days) == seq_len:
				new = scaler.fit_transform(np.array(prev_days))
				X.append(new)

	return np.array(X)


def getData(stockList,upThresh=5):
	# obtain target and features for each ticker
	# "bin" is short for binary
	scaler = MinMaxScaler() # Scaler used to standardize features
	#use = ['sma','ema','dema','adx','aroondown','aroonup','cci','cmo','macd','mfi','mom','roc','rsi','stoch', 'stochd','trix','obv','atr','wpr','cad','bin']
	lkback = [n for n in range(1,21)] # list of lookback distances

	X = []
	y = []
	for ticker in stockList:
		print(f'Getting {ticker} data.')
		time.sleep(1)
		df = yf.download(ticker,start=start_date,end=end_date,progress=False)
		print(df.isnull().values.any())

		use = []
		for item in lkback:
			df[f"shift{item}"] = np.log(df['Adj Close']/df['Adj Close'].shift(item))
			#if item == periodToPredict:
			use.append(f"shift{item}")
		use.append("bin")
		# df['sma'] = talib.SMA(df['Adj Close'], timeperiod=20)
		# df['ema'] = talib.EMA(df['Adj Close'], timeperiod=20)
		# df['dema'] = talib.DEMA(df['Adj Close'], timeperiod=20)
		# df['adx'] = talib.ADX(df['High'],df['Low'],df['Close'], timeperiod=14)
		# df['aroondown'], df['aroonup'] = talib.AROON(df['High'],df['Low'],timeperiod=14)
		# df['cci'] = talib.CCI(df['High'],df['Low'],df['Close'], timeperiod=14)
		# df['cmo'] = talib.CMO(df['Adj Close'], timeperiod=14)
		# df['macd'], _, _ = talib.MACDFIX(df['Adj Close'])
		# df['mfi'] = talib.MFI(df['High'],df['Low'],df['Close'],df['Volume'], timeperiod=14)
		# df['mom'] = talib.MOM(df['Adj Close'], timeperiod=10)
		# df['roc'] = talib.ROC(df['Adj Close'], timeperiod=10)
		# df['rsi'] = talib.RSI(df['Adj Close'], timeperiod=14)
		# df['stoch'], df['stochd'] = talib.STOCHF(df['High'],df['Low'],df['Close'])
		# df['trix'] = talib.TRIX(df['Adj Close'], timeperiod=30)
		# df['obv'] = talib.OBV(df['Adj Close'],df['Volume'])
		# df['atr'] = talib.ATR(df['High'],df['Low'],df['Close'], timeperiod=14)
		# df['wpr'] = talib.WILLR(df['High'],df['Low'],df['Close'], timeperiod=14)
		# df['cad'] = talib.ADOSC(df['High'],df['Low'],df['Close'],df['Volume'])


		df['pct_chg'] = (df['Adj Close'].shift(-periodToPredict)/df['Adj Close'] -1) * 100
		#df['targ'] = np.where(df['pct_chg'] > 0, 1, 0)
		# mask = (df['pct_chg'] < upThresh) & (df['pct_chg'] > -upThresh)

		# df['bin'][mask] = True
		# df['bin'][~mask] = False
		df['bin'] = np.where((df['pct_chg'] < upThresh), True, False)
		#df['bin'] = np.where(df['logdiff'] > (df['ma']+df['std']), False, True)
		df.dropna(inplace=True)

		seq_len = len(use)-1
		prev_days = deque(maxlen=seq_len)

		for i in df[use].values:
			# create (x,x) array along with target
			prev_days.append([n for n in i[:-1]])
			if len(prev_days) == seq_len:
				new = scaler.fit_transform(np.array(prev_days))
				X.append(new)
				y.append(i[-1])
	return np.array(X), np.array(y)
#getToday(stockList)

X, y = getData(stockList)

print(len(y))

oob_x = X[-50:] # out-of-bag data 
oob_y = y[-50:]
X = X[:-50]
y = y[:-50]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=874521)
#_, _, _, chg = train_test_split(df['targ'],df['pct_chg'],test_size=0.2,random_state=874521)
X_train = X_train[..., tf.newaxis] # resize
X_test = X_test[..., tf.newaxis]
oob_x = oob_x[..., tf.newaxis]

# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# oob_x = scaler.transform(oob_x)

normal_X_train = X_train[y_train] # data of one type to be trained upon
normal_X_test = X_test[y_test]
# for t in range(len(normal_X_train)):
# 	if np.shape(normal_X_train[t])[0] != 19 or np.shape(normal_X_train[t])[1] != 19:
# 		print(t)
# 		print("error")
# 		break

diff_X_train = X_train[~y_train] # other type to exclude
diff_X_test = X_test[~y_test]
print(len(diff_X_test))

ae = AutoEncoder()
opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
ae.compile(optimizer=opt, loss='mse')

history = ae.fit(normal_X_train,normal_X_train,
	epochs=40,
	batch_size=512,
	validation_data=(normal_X_test,normal_X_test),
	shuffle=True)

### Plot loss over epochs
# plt.plot(history.history["loss"], label="Training Loss")
# plt.plot(history.history["val_loss"], label="Validation Loss")
# plt.legend()
# plt.show()


### Plot reconstructed signal and error
# encoded_data = ae.encoder(up_X_test).numpy()
# decoded_data = ae.decoder(encoded_data).numpy()

# plt.plot(up_X_test[0], 'b')
# plt.plot(decoded_data[0], 'r')
# plt.fill_between(np.arange(len(feats)), decoded_data[0], up_X_test[0], color='lightcoral')
# plt.legend(labels=["Input", "Reconstruction", "Error"])
# plt.show()
def get_mae(original, reconstruction):
    # Returns the mean square error for each image in the array
    return np.mean((original - reconstruction)**2, axis=(1,2,3)) 
### Plot histogram of reconstruction error for up data
reconstructions = ae.predict(normal_X_test)
train_loss = get_mae(normal_X_test,reconstructions)#tf.reduce_mean(tf.abs(reconstructions - normal_X_test))
# print(len(train_loss))
# print(len(normal_X_test))
# plt.hist(train_loss, bins=50)
# plt.xlabel("Train loss")
# plt.ylabel("No of examples")
# plt.show()

threshold = np.mean(train_loss) + np.std(train_loss)
print("Threshold: ", threshold)

### Plot histogram of reconstruction error for down data
reconstructions = ae.predict(diff_X_test)
test_loss = get_mae(diff_X_test,reconstructions)

# plt.hist(test_loss, bins=50)
# plt.xlabel("Test loss")
# plt.ylabel("No of examples")
# plt.show()

def backtest(true,pred,pctchg,reverse=False):
	if reverse: # reverse boolean predictions
		import operator
		pred = list(map(operator.not_, pred))
	perf = []
	for a in range(len(true)):
		# loop through predictions to compare to actual results and append performances
		if true[a] and pred[a]:
			perf.append(pctchg[a])
		elif not true[a] and pred[a]:
			perf.append(pctchg[a])
		elif not true[a] and not pred[a]:
			perf.append(-pctchg[a])
		else:
			perf.append(-pctchg[a])
	return np.array(perf)

def predict(model, data, threshold):
	# return boolean if reconstruction loss is less than threshold
  reconstructions = model(data)
  loss = get_mae(data,reconstructions)
  return tf.math.less(loss, threshold)

from sklearn.metrics import accuracy_score, precision_score, recall_score

def print_stats(predictions, labels,prt=0):
	if prt == 1:
		rez = pd.DataFrame({'Predicted':np.array(predictions),'Actual':labels})
		print(rez)
		print("\nPredicted: ",Counter(np.array(predictions)))
		print("Actual: ",Counter(labels))
	print("Accuracy = {}".format(accuracy_score(labels, predictions)))
	print("Precision = {}".format(precision_score(labels, predictions)))
	print("Recall = {}".format(recall_score(labels, predictions)))

preds = predict(ae, X_test, threshold)
print_stats(preds, y_test)
#conf_matrix(y_test,preds)

#print(np.sum(backtest(y_test,preds,chg)))

predsnew = predict(ae, oob_x, threshold)
print_stats(predsnew, oob_y,prt=0)
#conf_matrix(oob_y,predsnew)

performance = backtest(oob_y,predsnew,oob_pctchg,reverse=True)
print("Performance: ",np.sum(performance))
print("Compounding: ", compounding(performance,1))
print(oob.tail())
#print(oob_x[-5:])