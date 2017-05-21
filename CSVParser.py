import csv
import numpy as np

class CSVParser:
	'Class for reading csv file and fetching parameters and labels'
	params = []
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
		self.params = nparrValues[:,0]
		self.labels = nparrValues[:,1]
		
	def getParams(self):
		return self.params

	def getLabels(self):
		return self.labels				
