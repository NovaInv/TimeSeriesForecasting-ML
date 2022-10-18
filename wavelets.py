# NovaInv August 3, 2022
# Random Forest Classification of binary one-step moves in time series
# using wavelets of lookback returns as features
# Reference: https://medium.com/engineer-quant/alphaai-using-machine-learning-to-predict-stocks-79c620f87e53

import yfinance as yf
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import pywt

df = yf.download('SPY',start='2017-01-01',end='2022-08-01',progress=False)
df['logdiff'] = np.log(df['Adj Close'].shift(-1)/df['Adj Close'])
df['targ'] = np.where(df['logdiff'] > 0, 1, 0)

lkback = [1,2,3,4,5,6,7,8,9,10,11,12,20,50,75,100,120,140,160,240,252]
feats = []
for item in lkback:
	df[f"lookback{item}"] = ((df['Adj Close']-df['Adj Close'].shift(item))/df['Adj Close'].shift(item))
	feats.append(f"lookback{item}")

df.dropna(inplace=True)

# loop over each feature and transform feature into inverse wavelet
for i in feats:
	ca, cd = pywt.dwt(df[i], wavelet="haar") # generate haar wavelet from feaure
	# coeffs = pywt.wavedec(df['logdiff'], "haar", level=8)
	# recon = pywt.waverec(coeffs,"haar")
	cat = pywt.threshold(ca, np.std(ca), mode="soft") # filter by std of wavelet  
	cdt = pywt.threshold(cd, np.std(cd), mode="soft") # filter by std of wavelet              
	tx = pywt.idwt(cat, cdt, "haar") # inverser discrete wavelet transform
	df[i] = tx # set feature from wavelet reconstruction

# use features on variet of Sci-Kit Learn classifiers
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
#clf = GradientBoostingClassifier(n_estimators=2000,random_state=777)
#clf = SGDClassifier(loss='squared_hinge',random_state=777)
clf = RandomForestClassifier(n_estimators=500,max_depth=15,max_features='sqrt',random_state=777, criterion='entropy',n_jobs=-1) # features from grid search
X = df[feats]
y = df['targ']

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False,random_state=777)

clf.fit(X_train,y_train)
# pred = clf.predict(X_test)
# print(accuracy_score(y_test,pred))
print(clf.score(X_test,y_test))
# featureArray = np.array(clf.feature_importances_)
# plt.barh(feats,featureArray)
# plt.xlim(np.min(featureArray),np.max(featureArray))
# plt.show()

#plt.plot(range(len(ca[-100:])),ca[-100:],label='ca')
# plt.plot(range(len(tx[-100:])),tx[-100:], label='tx')
# plt.plot(range(len(cd[-100:])),df['logdiff'].iloc[-100:], label='logdiff')
# plt.legend(loc="upper left")
# plt.show()