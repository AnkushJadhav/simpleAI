import sys
from SimpleLinearRegression import SimpleLinearRegression
csvFile = sys.argv[1]
predictParam = sys.argv[2]
linReg = SimpleLinearRegression()
params, labels = linReg.getData(csvFile)
B0, B1 = linReg.trainModel(params, labels)
predicted = linReg.predictLabel(predictParam, B0, B1)
print('predicted = ' + str(predicted))
rmse = linReg.getRMSE(params, labels, B0, B1)
print('rmse = ' + str(rmse))
