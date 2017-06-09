import sys
sys.path.append('..')
import math
import numpy as np
from CSVParser import CSVParser

class SimpleLogisticRegression:
	'A simple implementation of linear regression with RMSE'

	def getData(self, csvFile):
		dataStream = CSVParser(csvFile)
		dataStream.readFile()
		features = dataStream.getFeatures()
		labels = dataStream.getLabels()
		return (features, labels)

	def trainModel(self, params, labels):
		sumX = 0
		sumY = 0
		length = params.size
		for pos in range(0,length):
			sumX += int(params[pos])
			sumY += int(labels[pos])

		meanX = sumX/length
		meanY = sumY/length

		sumNum = 0
		sumDen = 0
		for pos in range(0,length):
			sumNum += ((int(params[pos]) - meanX)*(int(labels[pos]) - meanY))
			sumDen += math.pow((int(params[pos]) - meanX), 2)
		
		B1 = sumNum/sumDen
		B0 = meanY - (B1 * meanX)

		return (B0, B1)

	def predictLabel(self, predictParam, B0, B1):
		predictedLabel = (math.exp(B0 + (B1 * float(predictParam)))) / (1 + math.exp(B0 + (B1 * float(predictParam))))
		return predictedLabel

	def getRMSE(self, params, labels, B0, B1):
		length = params.size
		sumPred = 0
		for pos in range(0,length):
			sumPred += math.pow((float(self.predictLabel(params[pos], B0, B1)) - float(labels[pos])),2)
		
		rmse = math.sqrt(sumPred/length)
		return rmse
