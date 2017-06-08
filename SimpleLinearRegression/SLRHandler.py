import sys
from SimpleLinearRegression import SimpleLinearRegression
csvFile = 'test.csv'
predictFeature = sys.argv[1]
linReg = SimpleLinearRegression()
features, labels = linReg.getData(csvFile)
B0, B1 = linReg.trainModel(features, labels)
predicted = linReg.predictLabel(predictFeature, B0, B1)
print('predicted = ' + str(predicted))
rmse = linReg.getRMSE(features, labels, B0, B1)
print('rmse = ' + str(rmse))
