# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 08:09:53 2021

@author: hamem
"""

from pandas import read_csv
import numpy as np
from sklearn.model_selection import KFold
import os
from datetime import datetime
from matplotlib import pyplot as plt
from multiprocessing import Process
import matplotlib.patches as mpatches
import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.cr-oss_validation import cross_val_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.svm import SVC
#import xgboost as xgb
#from xgboost.sklearn import XGBClassifier
import random
import os
import numpy as np
from sklearn.metrics import roc_curve,auc, confusion_matrix
from matplotlib import pyplot as plt
from matplotlib import animation
from sklearn.ensemble import RandomForestClassifier

class Evaluator:

	def __init__(self,xtest,ytest,y_pred,LearningModel):
		self.xtest = xtest
		self.ytest = ytest
		self.y_pred = y_pred
		self.model = LearningModel

	def confusionMatrix(self):

		self.confusion_matrix = confusion_matrix(self.ytest, self.y_pred)
		#print self.ytest
		#print self.y_pred
	"""
	def getPerformanceMetrics(self):
		self.confusionMatrix()
		accuracy = (
					float((self.confusion_matrix[0][0]+self.confusion_matrix[1][1]))/
					(self.confusion_matrix[0][0]+self.confusion_matrix[0][1]+self.confusion_matrix[1][0]+self.confusion_matrix[1][1])
			)
		precision = (
					float((self.confusion_matrix[1][1]))/
					(self.confusion_matrix[1][1] + self.confusion_matrix[0][1])
			)
		recall = (
				float((self.confusion_matrix[1][1]))/
				(self.confusion_matrix[1][1]+self.confusion_matrix[1][0])
			)
		specificity = (
				float((self.confusion_matrix[0][0]))/
				(self.confusion_matrix[0][0] + self.confusion_matrix[0][1])
			)
		return accuracy, recall, precision, specificity
	"""


	def getPerformanceMetrics(self):

		flag = 0
		for i in range(len(self.ytest)):
			if self.ytest[i] != self.y_pred[i]:
				flag = 1
				break

		if flag == 0:
			if -1 in self.ytest:
				return len(self.ytest), 0, 0, 0
			else:
				return 0, 0, len(self.ytest), 0

		self.confusionMatrix()
		return self.confusion_matrix[0][0],self.confusion_matrix[1][0],self.confusion_matrix[1][1],self.confusion_matrix[0][1]

	def drawROC(self):

		base_dir = os.path.dirname
		abspath = os.path.abspath
		dir_name =  base_dir(base_dir(base_dir(abspath(__file__))))

		y_prob = self.model.predict_proba(self.xtest)

		true_probability_estimate = y_prob[:,1]

		fpr,tpr,threshold = roc_curve(self.ytest,true_probability_estimate)
		area = auc(fpr,tpr)
		plt.figure()
		plt.plot(fpr,tpr,linewidth = 2.0,label = "ROC curve (Area= %0.2f)" % area)
		plt.plot([0,1],[0,1],"r--")
		plt.xlabel("False Postive Rate")
		plt.ylabel("True Positive Rate")
		plt.legend(loc = "lower right")
		plt.show(block = False)

		#plt.savefig(savepath)
		#plt.close()

	def oob_vs_n_trees(self,max_trees,Xtrain, ytrain):

		# First set up the figure, the axis, and the plot element we want to animate
		print("")
		print("Number of Trees\t\tOOB Error Rate")
		fig = plt.figure()
		ax = plt.axes(xlim=(0, max_trees), ylim=(0,1))
		line, = ax.plot([], [], lw=2)

		# initialization function: plot the background of each frame
		def init():
		    line.set_data([], [])
		    return line,

		number_of_trees = range(2,max_trees + 1)
		oob_errors = []
		# animation function.  This is called sequentially
		def animate(i):

			model = RandomForestClassifier(warm_start = True, oob_score = True, n_estimators = i)
			model.fit(Xtrain,ytrain)
			oob_error = 1 - model.oob_score_
			oob_errors.append(oob_error)
			print("{}\t\t\t{}".format(i,oob_error))

			line.set_data(number_of_trees[:len(oob_errors)], oob_errors)
			return line,

		# call the animator.  blit=True means only re-draw the parts that have changed.
		anim = animation.FuncAnimation(fig, animate, init_func=init, frames=number_of_trees, interval=100, blit=True, repeat = False)
		plt.xlabel("Number of trees")
		plt.ylabel("OOB error")
		plt.show()





		# for i in xrange(2,max_trees + 1):
		# 	model = RandomForestClassifier(warm_start = True,
		# 		oob_score = True,
		# 		n_estimators = i)
		# 	model.fit(Xtrain,ytrain)
		# 	oob_error = 1 - model.oob_score_
		# 	oob_errors.append(oob_error)
		# 	print i,oob_error


	def plotClassificationResult(self):
		self.confusionMatrix()
		#x = [i + 3.0 for i in xrange(4)]
		x = [i + 3.0 for i in range(4)]
		xlabel = ["TP","FN","FP","TN"]
		plt.figure()
		plt.grid(True)
		plt.bar(x,self.confusion_matrix.reshape(-1), color= np.random.random((4,3)))
		#plt.xticks([i + 3.0 for i in xrange(4)],xlabel)
		plt.xticks([i + 3.0 for i in range(4)],xlabel)
		plt.show(block = False)


import numpy as np
from pandas.stats.moments import ewma as smoother

class DataPreprocessor:

	def __init__(self):
		pass

	def CustomSmoother(self,x,alpha):

		s0 = x[0]
		smoothed_statistic = [s0]
		n = x.shape[0]
		for i in xrange(1,n):
			s1 = alpha * x[i] + (1 - alpha) * s0
			smoothed_statistic.append(s1)
			s0 = s1
		smoothed_statistic = np.array(smoothed_statistic)
		return smoothed_statistic

	def PandaSmoother(self,x):

		return smoother(x, span = 20)
np.set_printoptions(precision=2)

def getPerformanceStats(TP, TN, FP, FN):
	#print TP, TN, FP, FN
	accuracy = ((TP + TN)/(TP + TN + FP + FN))*100

	try:
		recall =  TP/(TP+FN)
	except:
		recall = -999

	try:
		precision = TP/(TP + FP)
		print(TP)
		print(FP)
	except:
		precision = -999

	try:
		specificity = TN/(TN + FP)
	except:
		specificity = -999

	try:
		fscore = 2*(precision * recall)/(precision + recall)
	except:
		fscore = -999
	return ("%.2f" % accuracy), ("%.2f" % recall), ("%.2f" % precision), ("%.2f" % specificity), ("%.2f" % fscore)

def getData(CSVFile):

	smoother = DataPreprocessor()
	data = read_csv(CSVFile)
	data = data[::-1] # reverse
	ohclv_data = np.c_[data['Open'],
					   data['High'],
					   data['Low'],
					   data['Close'],
					   data['Volume']]

	smoothened_ohclv_data = smoother.PandaSmoother(ohclv_data)
	return  smoothened_ohclv_data, np.array(data["Close"]), list(data["Date"])

def getTechnicalIndicators(X,d):

	RSI = getRSI(X[:,3])
	StochasticOscillator = getStochasticOscillator(X)
	Williams = getWilliams(X)


	MACD = getMACD(X[:,3])
	PROC = getPriceRateOfChange(X[:,3],d)
	OBV = getOnBalanceVolume(X)

	min_len = min(len(RSI),
				  len(StochasticOscillator),
				  len(Williams),
				  len(MACD),
				  len(PROC),
				  len(OBV))

	RSI = RSI[len(RSI) - min_len:]
	StochasticOscillator = StochasticOscillator[len(StochasticOscillator) - min_len:]
	Williams = Williams[len(Williams) - min_len: ]
	MACD = MACD[len(MACD) - min_len:]
	PROC = PROC[len(PROC) - min_len:]
	OBV = OBV[len(OBV) - min_len:]


	feature_matrix = np.c_[RSI[:,0],
						   StochasticOscillator[:,0],
						   Williams[:,0],
						   MACD[:,0],
						   PROC[:,0],
						   OBV[:,0]]

	return feature_matrix

def prepareData(X,close,date,d):

	feature_matrix = getTechnicalIndicators(X,d)

	number_of_samples = feature_matrix.shape[0]
	date = date[len(date) - number_of_samples:]
	close = close[len(close) - number_of_samples:]

	#y0 = feature_matrix[:,-1][ :number_of_samples-d]
	#y1 = feature_matrix[:,-1][d:]

	y0 = close[:number_of_samples - d]
	y1 = close[d:]

	feature_matrix_1 = feature_matrix[:number_of_samples-d]
	feature_matrix_2 = feature_matrix[number_of_samples - 1000:]
	date = date[number_of_samples - 1000:]



	#closeplot = feature_matrix[:,-1][number_of_samples - 1000:]
	closeplot = close[number_of_samples - 1000:]
	y = np.sign(y1 - y0)

	feature_matrix_1 = feature_matrix_1[:, range(6)]

	return feature_matrix_1,y,feature_matrix_2[:,range(6)],closeplot,date



def plotTradingStrategy(model, xplot, closeplot, Trading_Day,date):

	colorMap = {-1.0:"r",1.0:"b",0.0:"y"}
	tradeMap = {-1.0:"Sell",1.0:"Buy",0.0:"Buy"}
	plt.figure()
	plt.plot(closeplot, c = "g")
	#x = [xplot[i] for i in xrange(0,len(xplot),Trading_Day)]
	#y = [closeplot[i] for i in xrange(0, len(closeplot),Trading_Day)]
	x = [xplot[i] for i in range(0,len(xplot),Trading_Day)]
	y = [closeplot[i] for i in range(0, len(closeplot),Trading_Day)]
	y_pred = model.predict(x)

	#c = [colorMap[y_pred[i]] for i in xrange(len(y_pred))]
	c = [colorMap[y_pred[i]] for i in range(len(y_pred))]

	#df = pd.DataFrame(np.c_[[ i+1 for i in xrange(0, len(xplot),Trading_Day)], x, y, [tradeMap[y_pred[i]] for i in xrange(len(y_pred)) ]],
	df = pd.DataFrame(np.c_[[ i+1 for i in range(0, len(xplot),Trading_Day)], x, y, [tradeMap[y_pred[i]] for i in range(len(y_pred)) ]],
   			columns = ["Day","RSI","Stochastic Oscillator","Williams","MACD","Price Rate Of Change","On Balance Volume","Close","Buy/Sell"])
	df.to_csv("AAPLBuySellTradePoints.csv",index = False)


	#plt.scatter([i for i in xrange(0,len(xplot),Trading_Day)],y, c = c)
	plt.scatter([i for i in range(0,len(xplot),Trading_Day)],y, c = c)
	#plt.xticks([i for i in xrange(0,len(xplot),Trading_Day)],[date[i] for i in xrange(0,len(xplot),Trading_Day)])
	red_patch = mpatches.Patch(color='red', label='Sell')
	blue_patch = mpatches.Patch(color = "blue", label = "Buy")
	plt.legend(handles = [red_patch,blue_patch])
	plt.xlabel("Time")
	plt.ylabel("Closing price")
	#plt.title("Trading strategy for {} days trading window".format(Trading_Day))
	plt.title("Trading indications for {} days trading window".format(Trading_Day))
	plt.savefig("TradingStrategy.png")
	plt.show(block = False)


def main(stock_symbol,Trading_Day, classifier_choice):

	# fetcher = DataFetcher()

	# fetch_result = fetcher.getHistoricalData(stock_symbol)
	# if fetch_result == -1:
	# 	raise Exception("NO INTERNET CONNECTIVITY OR INVALID STOCK SYMBOL")

	dir_name = os.path.dirname(os.path.abspath(__file__))
	filename = stock_symbol+".csv"
	CSVFile = os.path.join(dir_name,"Dataset",filename)

	ohclv_data, close, date= getData(CSVFile)

	#current_data = regression(ohclv_data)
	#ohclv_data.append(current_data)

	ohclv_data = np.array(ohclv_data)

	X,y,xplot,closeplot,dateplot = prepareData(ohclv_data, close, date, Trading_Day)

	y[y == 0] = 1

	TP = 0.0
	TN = 0.0
	FP = 0.0
	FN = 0.0
	NUM_ITER = 1
	pred_prob = []
	true_lbls = []
	scr_list = []

	for iteration in range (0, NUM_ITER):
		Xtrain,Xtest,ytrain,ytest = train_test_split(X,y, random_state = 0)
		#dummies, inserted by Suryo:
		#print "Length of training set:", len(Xtrain)
		#print "Length of test set:", len(Xtest)

		if classifier_choice == 'RF':
			#model = RandomForestClassifier(n_estimators = 100,criterion = "gini", random_state = random.randint(1,12345678))
			model = SVC(kernel='rbf', tol=0.01, max_iter=50000, probability=True)

			model.fit(Xtrain, ytrain)
			print('we here', stock_symbol)
			y_pred = model.predict(Xtest)


		Eval = Evaluator(Xtest,ytest,y_pred,model)
		tn, fn, tp, fp = Eval.getPerformanceMetrics()
		TN += tn
		FN += fn
		TP += tp
		FP += fp
		#================================
		#y_score = model.fit(Xtrain, ytrain).decision_function(test_data)
		y_prob = model.predict_proba(Xtest)
		#print(y_prob)
		for i in range(len(y_prob)):
			if ytest[i] == -1.0:
				true_lbls.append(0)
			else:
				true_lbls.append(1)
			try:
				pred_prob.append(y_prob[i][1])
			except:
				pred_prob.append(y_prob[i][0]) #There's an error probably due to perfect classification
			#scr_list.append(y_score[i])

	accuracy, recall, precision, specificity, fscore = getPerformanceStats(TP, TN, FP, FN)
	for i in range(len(ytest)):
		if ytest[i] == -1.0:
			ytest[i] = 0
		if y_pred[i] == -1.0:
			y_pred[i] = 0
	try:
		auc = roc_auc_score(true_lbls,pred_prob)
		fpr, tpr, thresholds = roc_curve(true_lbls, pred_prob, pos_label=1)
		#print(fpr)
		#print(tpr)
		plt.figure()
		lw = 2
		plt.plot(fpr, tpr, color='darkorange',
         	lw=lw, label='ROC curve (area = %0.2f)' % auc)
		#plt.plot(fpr, tpr, color='darkorange')
		plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title(str(stock_symbol)+'-'+str(Trading_Day)+' day-SVM')
		plt.legend(loc="lower right")
		plt.savefig('rocs/'+str(stock_symbol)+'-'+str(Trading_Day)+'-'+'SVM.png')
		plt.clf()
	except:
		"Can't plot ROC"
		auc = -999

	b_score = brier_score_loss(ytest,y_pred)

	print(stock_symbol, '&', Trading_Day, '&', accuracy, '&', recall, '&',
			precision, '&', specificity, '&', fscore, '&', '%0.2f' % b_score,
			'&', '%0.2f' % auc, '\\\\')

	"""
	print ""
	print "Accuracy:",accuracy
	print "Recall:",recall
	print "Precision:",precision
	print "Specificity:",specificity
	print "F-Score:",fscore
	"""
	Eval.plotClassificationResult()
	Eval.drawROC()
	plotTradingStrategy(model,xplot,closeplot,Trading_Day,dateplot)
	#c = raw_input("Press y to generate OOB vs Number of estimators graph:")
	#if c == "y" or c == "Y":
	#Eval.oob_vs_n_trees(100,Xtrain,ytrain)
	#"""

	# raw_input("Press enter to genereate OOB vs Number of estimators graph:")
	# p.start()
	# print "LOL"
	# p.join()


"""
stock_symbol = raw_input("Enter the stock_symbol (AAPL, AMS, AMZN, FB, MSFT, NKE, SNE, TATA, TWTR, TYO): ")
Trading_Day = input("Enter the trading window: ")
classifier_choice = raw_input("Enter the classifier of choice (RF, XGB): ")
if classifier_choice not in ['RF', 'XGB']:
	print 'Invalid classifier.'
	exit()
main(stock_symbol.upper(),Trading_Day, classifier_choice.upper())
"""
#COMPANIES = ['AAPL', 'AMS', 'AMZN', 'FB', 'MSFT', 'NKE', 'SNE', 'TATA', 'TWTR', 'TYO']
#COMPANIES = ['novartis', 'cipla', 'pfizer', 'roche']
COMPANIES = ['AAPL']
#COMPANIES = ['ROG', 'NOVN', 'PFE', 'CIPLA']
TRADING_BINS = [3, 5, 10, 15, 30, 60, 90]
CLASSIFIERS = ['RF']
COMPANIES = ['AMZN', 'AAPL', 'AMS']
#COMPANIES = ['AAPL', 'AMS', 'AMZN', 'FB', 'MSFT',
#			'NKE', 'SNE', 'TATA', 'TWTR', 'TYO',
#			'novartis', 'cipla', 'pfizer', 'roche']

for classifier_choice in CLASSIFIERS:
	print("---------------------")
	print("")
	print('RESULTS FOR', classifier_choice)
	print("")

	for stock_symbol in COMPANIES:
		for Trading_Day in TRADING_BINS:
			#main(stock_symbol.upper(),Trading_Day, classifier_choice.upper())
			main(stock_symbol,Trading_Day, classifier_choice.upper())