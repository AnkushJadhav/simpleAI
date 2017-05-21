import csv
import numpy as np

class CSVParser:
	'Class for reading csv file and fetching features and labels'
	features = []
	labels = []
	csvFile = ''

	def __init__(self, csvFile):
		self.csvFile = csvFile

	def readFile(self):
		values = [];
		with open(self.csvFile,'r') as csvfile:
			csvstream = csv.reader(csvfile, delimiter=',')
			for row in csvstream:
				values.append(row)
		nparrValues = np.array(values)
		self.features = nparrValues[:,0]
		self.labels = nparrValues[:,1]
		
	def getFeatures(self):
		return self.features

	def getLabels(self):
		return self.labels				
