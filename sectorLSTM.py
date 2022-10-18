# NovaInv April 21, 2021
# Long Short Term Memory Neural Network
# Train and predict based on asset price based features or sector performances
# Reference: https://pythonprogramming.net/crypto-rnn-model-deep-learning-python-tensorflow-keras/

import pandas as pd
import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
import time
from collections import Counter
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import yfinance as yf
import talib
###
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
#only works on sublime build python3.6
def equalize(data,r=777):
    # resample and balance data between 0 and 1 targets
    sample_in = int(min(list(dict(Counter(data['targ'])).values()))-1)
    df_1 = data[data['targ'] == 0]
    df_2 = data[data['targ'] == 1]
    df_1 =df_1.sample(n=sample_in, random_state = r)
    df_2 =df_2.sample(n=sample_in, random_state = r)
    data = df_1.append(df_2)
    data = data.sample(frac=1, random_state=r)
    data_targ = data['targ']
    data.drop(['targ'],axis=1,inplace=True)
    return data, data_targ
def process(df,targgg):
	# sequential_data = []  # this is a list that will CONTAIN the sequences
	# prev_days = deque(maxlen=SEQ_LEN) # variable used to pop and add variables to create a consistent sliding window

	# for i in df.values:  # iterate over the values
	# 	prev_days.append([n for n in i[:-1]])  # store all but the target
	# 	if len(prev_days) == SEQ_LEN:
	# 		sequential_data.append(np.array(prev_days))
    df = pd.DataFrame(scaler.transform(df))
    sequential_data = []  
    prev_days = deque(maxlen=SEQ_LEN)  
    targvals = targgg.values
    s = 0
    for i in df.values:  # iterate over the values
        prev_days.append([n for n in i])  # store all but the target
        if len(prev_days) == SEQ_LEN: 
            sequential_data.append([np.array(prev_days), targvals[s]]) 
        s += 1

    X = []
    y = []

    for seq, target in sequential_data:  # going over our new sequential data
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets

    
    #return np.array(sequential_data)
    return np.array(X), np.array(y).astype(int)


scaler = StandardScaler()
def preprocess_df(df,targg,val):

    if val==1:
        
        df = pd.DataFrame(scaler.fit_transform(df))
    else:
        df = pd.DataFrame(scaler.transform(df))
    sequential_data = []  # this is a list that will CONTAIN the sequences
    prev_days = deque(maxlen=SEQ_LEN)  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in
    targvals = targg.values
    s = 0
    for i in df.values:  # iterate over the values
        prev_days.append([n for n in i])  # store all but the target
        if len(prev_days) == SEQ_LEN:  # make sure we have 60 sequences!
            sequential_data.append([np.array(prev_days), targvals[s]])  # append those bad boys!
        s += 1

    random.shuffle(sequential_data)  # shuffle for good measure.

    buys = []  # list that will store our buy sequences and targets
    sells = []  # list that will store our sell sequences and targets

    for seq, target in sequential_data:  # iterate over the sequential data
        if target == 0:  # if it's a "not buy"
            sells.append([seq, target])  # append to sells list
        elif target == 1:  # otherwise if the target is a 1...
            buys.append([seq, target])  # it's a buy!

    random.shuffle(buys)  # shuffle the buys
    random.shuffle(sells)  # shuffle the sells!

    lower = min(len(buys), len(sells))  # what's the shorter length?

    buys = buys[:lower]  # make sure both lists are only up to the shortest length.
    sells = sells[:lower]  # make sure both lists are only up to the shortest length.

    sequential_data = buys+sells  # add them together
    random.shuffle(sequential_data)  # another shuffle, so the model doesn't get confused with all 1 class then the other.

    X = []
    y = []

    for seq, target in sequential_data:  # going over our new sequential data
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets/labels (buys vs sell/notbuy)

    

    return np.array(X), np.array(y).astype(int)

SEQ_LEN = 10  # how long of a preceeding sequence to collect for RNN
              #SEQ_LEN needs to be less than 0.05 of data length
EPOCHS = 40  # how many passes through our data
BATCH_SIZE = 256  # how many batches? Try smaller batch if you're getting OOM (out of memory) errors.
val_size = 0.2
tf.random.set_seed(789542)
useOOB = False
#main_df = pd.read_csv("Datasets/Spy_classify_lookbacks.csv")
start_date = '2014-01-23'
end_date = '2022-09-03'
ticker = 'SPY'
periodToPredict = 1 # how far ahead to predict

# list for lookback distance
lkback = [1,2,3,4,5,6,7,8,9,10,11,12,20,50,75,100,120,140,160,240,252]
feats = []
main_df = yf.download(ticker,start=start_date,end=end_date,progress=False) ### using yfinance
for item in lkback:
    # loop through and genrate return features
    main_df[f"lookback{item}"] = ((main_df['Adj Close']-main_df['Adj Close'].shift(item))/main_df['Adj Close'].shift(item))
    feats.append(f"lookback{item}")

main_df['pct_chg'] = ((main_df['Adj Close']-main_df['Adj Close'].shift(periodToPredict))/main_df['Adj Close'].shift(periodToPredict)).shift(-periodToPredict)*100
main_df['targ'] = np.where(main_df['pct_chg'] > 0, 1, 0)
main_df.drop(['Open','High','Low','Close','Adj Close','Volume','pct_chg'],axis=1,inplace=True)
main_df.dropna(inplace=True)
# oob = main_df[-50:].copy(deep=True) # out-of-bag array
# main_df = main_df[:-50]
main_df = pd.read_csv("Datasets/spyWithIndic.csv") # data from csv file
main_df.drop(['Date','pct_chg'],axis=1,inplace=True)
# print(main_df.head())

targ_df = main_df['targ'].copy(deep=True)
main_df.drop(['targ'],axis=1,inplace=True)
if useOOB: # use out-of-bag data?
    oob = main_df[-50:].copy(deep=True)
    oob_targ = targ_df[-50:].copy(deep=True)
    main_df = main_df[:-50]
    targ_df = targ_df[:-50]

# train_test_split data
sect_x = main_df.head(int(len(main_df)*(1-val_size)))
sect_targ = targ_df.head(int(len(main_df)*(1-val_size)))
train_x, train_y = preprocess_df(sect_x,sect_targ,val=1)
#train_y = targ_df[SEQ_LEN:len(train_x)+SEQ_LEN].to_numpy()
# print(train_x)

targ_x = main_df.tail(int(len(main_df)*(val_size)))
targ_y = targ_df.tail(int(len(main_df)*(val_size)))
validation_x, validation_y = preprocess_df(targ_x,targ_y,val=2)
print(validation_y)
print(validation_x.shape)
#validation_y = targ_df[SEQ_LEN:len(validation_x)+SEQ_LEN].to_numpy()

print(f"train data: {len(train_x)} validation: {len(validation_x)}")
print(train_x.shape[1:])
#print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
#print(f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.model_selection import train_test_split
# df = pd.read_csv("Datasets/spyWithIndic.csv")
# df.set_index("Date",inplace=True)

# pctChg = df['pct_chg']
# targ = df['targ']
# df.drop(['pct_chg'],axis=1,inplace=True)
# trainLen = 1400
# testLen = 100
# X, y = equalize(df)
# X_train, X_test, train_y, validation_y = train_test_split(X,y,test_size=0.2,shuffle=False)
# scaler = MinMaxScaler()
# train_x = scaler.fit_transform(X_train)
# validation_x = scaler.transform(X_test)
# train_x = np.expand_dims(np.array(train_x),1)
# validation_x = np.expand_dims(np.array(validation_x),1)
#validation_y = validation_y.reshape(1,len(validation_y),1)
#input_shape=(train_x.shape[1:])
############# Neural Network Model ###############
model = Sequential()
model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.15)) 
model.add(BatchNormalization())

model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.15))
model.add(BatchNormalization())

model.add(LSTM(64))
model.add(Dropout(0.15))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.15))

model.add(Dense(2, activation='softmax'))


opt = tf.keras.optimizers.Adamax(learning_rate=0.001, decay=1e-6)

# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

#tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

#filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
#checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')) # saves only the best ones
# train_y = np.array(train_y)
# validation_y = np.array(validation_y)
# Train model
history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS
    #callbacks=[tensorboard, checkpoint],
)

# Score model
score = model.evaluate(validation_x, validation_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# test_days = deque(maxlen=SEQ_LEN)
# test_data = []
# for i in end_df[feats].values:  # iterate over the values
#     test_days.append([n for n in i])  # store all but the target
#     if len(test_days) == SEQ_LEN:  # make sure we have 60 sequences!
#         test_data.append(np.array(test_days))
# print(test_data[3])
# print(np.array(test_data).shape)
# for p in range(len(test_data)):
#     this  = np.array(test_data[p])
#     that = np.reshape(this,(1,SEQ_LEN,len(feats)))
#     predictions = model.predict(that)
#     print(predictions)
# list to store and retrieve backtest variables
pred_list = []
probas = []
actual_list = []
#predictions = model.predict(validation_x)
for p in range(validation_x.shape[0]):
    # loop through LSTM predictions
    # this  = np.array(test_data[p])
    that = np.reshape(validation_x[p],(1,SEQ_LEN,validation_x.shape[2]))
    predictions = model.predict(that,verbose=0)
    #probas.append(predictions)
    if predictions[0][1] > 0.5:
        pred_list.append(1)
        probas.append(predictions)
        actual_list.append(validation_y[p])
    elif predictions[0][0] > 0.5:
        pred_list.append(0)
        probas.append(predictions)
        actual_list.append(validation_y[p])
    else:
        pass
    # if p%10==0:
    #     print(predictions,validation_y[p])
# print backtest statistics 
print("Predicted: ",Counter(pred_list))
print("Actual: ",Counter(actual_list))
print("Accuracy: ",accuracy_score(actual_list,pred_list))
rez = pd.DataFrame({'Probas':probas,'Predicted':pred_list,'Actual':actual_list})
print(rez) # print results 

# backtest the out-of-bag array
if useOOB:
    oob_x, oob_y = process(oob,oob_targ)

    pred_list = []
    probas = []
    actual_list = []
    #predictions = model.predict(validation_x)
    for p in range(oob_x.shape[0]):
        # this  = np.array(test_data[p])
        that = np.reshape(oob_x[p],(1,SEQ_LEN,oob_x.shape[2]))
        predictions = model.predict(that,verbose=0)
        #probas.append(predictions)
        if predictions[0][1] > 0.55:
            pred_list.append(1)
            probas.append(predictions)
            actual_list.append(oob_y[p])
        elif predictions[0][0] > 0.55:
            pred_list.append(0)
            probas.append(predictions)
            actual_list.append(oob_y[p])
        else:
            pass
        # if p%10==0:
        #     print(predictions,validation_y[p])
    print("Predicted: ",Counter(pred_list))
    print("Actual: ",Counter(actual_list))
    print("Accuracy: ",accuracy_score(actual_list,pred_list))
    rez = pd.DataFrame({'Probas':probas,'Predicted':pred_list,'Actual':actual_list})
    print(rez)

    print(oob.tail(12))
    print(oob_x[-1])