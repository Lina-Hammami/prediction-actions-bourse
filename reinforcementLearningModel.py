# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 17:44:37 2021

@author: hamem
"""



from agent.agent import Agent
from functions import *
import sys

if len(sys.argv) != 4:
	print "Usage: python train.py [stock] [window] [episodes]"
	exit()

stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

agent = Agent(window_size)
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32

for e in xrange(episode_count + 1):
	print "Episode " + str(e) + "/" + str(episode_count)
	state = getState(data, 0, window_size + 1)

	total_profit = 0
	agent.inventory = []

	for t in xrange(l):
		action = agent.act(state)

		# sit
		next_state = getState(data, t + 1, window_size + 1)
		reward = 0

		if action == 1: # buy
			agent.inventory.append(data[t])
			print "Buy: " + formatPrice(data[t])

		elif action == 2 and len(agent.inventory) > 0: # sell
			bought_price = agent.inventory.pop(0)
			reward = max(data[t] - bought_price, 0)
			total_profit += data[t] - bought_price
			print "Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price)

		done = True if t == l - 1 else False
		agent.memory.append((state, action, reward, next_state, done))
		state = next_state

		if done:
			print "--------------------------------"
			print "Total Profit: " + formatPrice(total_profit)
			print "--------------------------------"

		if len(agent.memory) > batch_size:
			agent.expReplay(batch_size)

	if e % 10 == 0:
		agent.model.save("models/model_ep" + str(e))


### Fonctions *******


import numpy as np
import math

# prints formatted price
def formatPrice(n):
	return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

# returns the vector containing stock data from a fixed file
def getStockDataVec(key):
	vec = []
	lines = open("data/" + key + ".csv", "r").read().splitlines()

	for line in lines[1:]:
		vec.append(float(line.split(",")[4]))

	return vec

# returns the sigmoid
def sigmoid(x):
	return 1 / (1 + math.exp(-x))

# returns an an n-day state representation ending at time t
def getState(data, t, n):
	d = t - n + 1
	block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
	res = []
	for i in xrange(n - 1):
		res.append(sigmoid(block[i + 1] - block[i]))

	return np.array([res])



## Evaluation **********

import keras
from keras.models import load_model

from agent.agent import Agent
from functions import *
import sys

if len(sys.argv) != 3:
	print "Usage: python evaluate.py [stock] [model]"
	exit()

stock_name, model_name = sys.argv[1], sys.argv[2]
model = load_model("models/" + model_name)
window_size = model.layers[0].input.shape.as_list()[1]

agent = Agent(window_size, True, model_name)
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32

state = getState(data, 0, window_size + 1)
total_profit = 0
agent.inventory = []

for t in xrange(l):
	action = agent.act(state)

	# sit
	next_state = getState(data, t + 1, window_size + 1)
	reward = 0

	if action == 1: # buy
		agent.inventory.append(data[t])
		print "Buy: " + formatPrice(data[t])

	elif action == 2 and len(agent.inventory) > 0: # sell
		bought_price = agent.inventory.pop(0)
		reward = max(data[t] - bought_price, 0)
		total_profit += data[t] - bought_price
		print "Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price)

	done = True if t == l - 1 else False
	agent.memory.append((state, action, reward, next_state, done))
	state = next_state

	if done:
		print "--------------------------------"
		print stock_name + " Total Profit: " + formatPrice(total_profit)
		print "--------------------------------"
