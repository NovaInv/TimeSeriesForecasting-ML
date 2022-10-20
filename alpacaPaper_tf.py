# NovaInv September 20, 2022
# Use AutoEncode reconstruction for daily trading signals
# on paper trading account. Currently, model is updated every two months.
# Reference: https://www.tensorflow.org/tutorials/generative/autoencoder

from alpaca_trade_api.rest import REST, TimeFrame
import numpy as np
import pandas as pd
import yfinance as yf 
from datetime import date, timedelta, datetime
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import layers
#import schedule
import math
import config
import time
import json
import joblib

class AutoEncoder(Model):
	def __init__(self,latent_dim,last_dim):
	    super(AutoEncoder, self).__init__()
	    self.last_dim = last_dim
	    self.latent_dim = latent_dim
	    self.encoder = tf.keras.Sequential([
	      layers.Dense(32, activation="relu"),
	      layers.Dense(16, activation="relu"),
	      layers.Dense(latent_dim, activation="relu")])

	    self.decoder = tf.keras.Sequential([
	      layers.Dense(16, activation="relu"),
	      layers.Dense(32, activation="relu"),
	      layers.Dense(last_dim, activation="sigmoid")])

	def call(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded

cash_displacement = 5  #cash to remove from cash avaliable to avoid using too much
r = 42
pending_positions = {}
position_counter = {}
position_ticker = {}
logfile = 'stocktrade.log'
pendinglog = 'stockpendingpositions.log'
counterlog = 'stockcounter.log'
tickerlog = 'stockticker.log'
API_KEY = config.stock_API_KEY
SECRET_KEY = config.stock_SECRET_KEY
alpaca = REST(API_KEY, SECRET_KEY, 'https://paper-api.alpaca.markets')
ticker = 'SPY'
longTicker = 'SPXL'
shortTicker = 'SPXS'
periodToPredict = 5

def prediction():
	##########  Variables  ##########
	today = date.today()
	start_date = today - timedelta(days=100)
	end_date = today + timedelta(days=1)

	df = yf.download(ticker,start=start_date,end=end_date,progress=False)
	df['logdiff'] = np.log(df['Adj Close']/df['Adj Close'].shift(periodToPredict)) #log returns

	lkback = [n for n in range(0,40)] #list for lookback distances
	feats = []
	for item in lkback:
		df[f"shift{item}"] = df['logdiff'].shift(item)
		feats.append(f"shift{item}")

	scaler = joblib.load('Models/standardScaler.gz') #from previous testing
	ae = load_model('Models/autoencoder_model9.tf') #from previous testing
	threshold = 1.2291504800457806 #from previous testing

	use = df[feats].iloc[-2:] #grab last two periods to use correct array shape
	use = scaler.transform(use)

	reconstruction = ae.predict(use,verbose=0)
	loss = tf.keras.losses.mae(reconstruction, use)
	out = tf.math.less(loss, threshold)[-1] #only want last period for decision

	return not out #inverse prediction from testing


def buy_order(ticker_):
	today = date.today()
	startt = today - timedelta(days=5)
	endd = today + timedelta(days=1)
	try:
		acct = alpaca.get_account() #get account data
		cash_avaliable = float(acct.cash)/(periodToPredict-len(pending_positions.keys()))-cash_displacement # how much cash avaliable for next trade
		recent_price = yf.download(ticker_,start=startt,end=endd,progress=False)['Adj Close']#get price from response
		buy_quantity = math.floor(cash_avaliable/float(recent_price[-1])) # get quantity for buy based on cash avaliable and recent price
		order = alpaca.submit_order(ticker_, buy_quantity, 'buy','market') #purchase

		#messages to log
		messagefile1 = f"\nBuy: {ticker_} qty: {buy_quantity} cash used: {cash_avaliable} at {datetime.now()} "
		messagefile2 = f"Order ID: {order.id}"
		#messagefile3 = f"\nCurrent Equity: {acct.last_equity}"
		with open(logfile, 'a') as f:
			f.write(messagefile1)
			f.write(messagefile2)
		
		return order

	except Exception as e:
		#log error
		with open(logfile, 'a') as f:
			f.write(f"\n{datetime.now()} ")
			f.write(str(e))
		#return nothing basically
		return 0

def job():
	global pending_positions
	global position_counter
	global position_ticker
	#print(f"Ran at: {datetime.now()}")
	order_ids_to_remove = []  # temporary list to record orders to be removed

	### Grabs dictionaries from log files
	with open(pendinglog,'r') as o:
		respond = o.read()
		if len(respond) > 0:
			pending_positions = json.loads(respond)

	with open(counterlog,'r') as o:
		respond = o.read()
		if len(respond) > 0:
			position_counter = json.loads(respond)

	with open(tickerlog,'r') as o:
		respond = o.read()
		if len(respond) > 0:
			position_ticker = json.loads(respond)
	#########

	activeMarket = alpaca.get_clock()
	if len(pending_positions.keys()) < periodToPredict and activeMarket.is_open == True: #if we still have more orders to go and market is open
		probability = prediction() #get prediction
		#print(probability)
		if probability: #if buy signal

			if len(position_ticker.keys()) == periodToPredict:
				#check to see if there is already a long position to restart
				for order_id in position_ticker:
					if position_ticker[order_id] == longTicker: #check each position for longTicker
						position_counter[order_id] == 0 #reset counter back to zero simulating new position
						break #stop the search for other long positions
				else:
					order = buy_order(longTicker) #execute buy order
					if order != 0: #if no error with the trade
						pending_positions[order.id] = order #store in pending position
						position_counter[order.id] = 0 # store in counter queue
						position_ticker[order.id] = longTicker


			else: #not full number of positions
				order = buy_order(longTicker) #execute buy order
				if order != 0: #if no error with the trade
					pending_positions[order.id] = order #store in pending position
					position_counter[order.id] = 0 # store in counter queue
					position_ticker[order.id] = longTicker


		if not probability: #if short signal
			if len(position_ticker.keys()) == periodToPredict:
				#check to see if there is already a short position to restart
				for order_id in position_ticker:
					if position_ticker[order_id] == shortTicker: #check each position for shortTicker
						position_counter[order_id] == 0 #reset counter back to zero simulating new position
						break #stop the search for other long positions
				else:
					order = buy_order(shortTicker) #execute buy order
					if order != 0: #if no error with the trade
						pending_positions[order.id] = order #store in pending position
						position_counter[order.id] = 0 # store in counter queue
						position_ticker[order.id] = shortTicker

			else: #not full number of positions
				order = buy_order(shortTicker) #execute buy order
				if order != 0: #if no error with the trade
					pending_positions[order.id] = order #store in pending position
					position_counter[order.id] = 0 # store in counter queue
					position_ticker[order.id] = shortTicker
			

	if len(position_counter.keys()) > 0 and activeMarket.is_open == True: # if there are positions in queue

		for order_id in position_counter: # loop through all positions

			if position_counter[order_id] == periodToPredict: # if position has reached its timelimit
				sell_quantity = float(pending_positions[order_id]['_raw']['qty'])
				whichTicker = position_ticker[order_id]
				sell_order = alpaca.submit_order(whichTicker,sell_quantity,'sell','market') # sell position
				order_ids_to_remove.append(order_id) # add to list to remove

				#messages to log
				messagefile1 = f"\nSell: {ticker} qty: {sell_quantity} at {datetime.now()}"
				messagefile2 = f"Order ID: {order_id}"
				with open(logfile, 'a') as f:
					f.write(messagefile1)
					f.write(messagefile2)
			
			else:
				position_counter[order_id] += 1  # add 1 to counter


	for order_id in order_ids_to_remove: # loop through list to remove
		del pending_positions[order_id]
		del position_counter[order_id]
		del position_ticker[order_id]

	### Write over log files with updated dictionaries
	with open(pendinglog,'w') as o:
		o.write(json.dumps(pending_positions, default=vars))

	with open(counterlog,'w') as o:
		o.write(json.dumps(position_counter))

	with open(tickerlog,'w') as o:
		o.write(json.dumps(position_ticker))

job()
