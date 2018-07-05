########################################
# Spam Filtering using Naive Bayes 	   #
#									   #
# LEE, Kristine-Clair                  #
# CS180                                #
########################################

import sys
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from bs4 import BeautifulSoup
import os, os.path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.metrics import accuracy_score, precision_score,f1_score
import numpy as np
import csv

np.set_printoptions(threshold=np.nan)
words = set(nltk.corpus.words.words())
stopWords = set(stopwords.words('english'))
ps = PorterStemmer()

###########################################################################
# Traverse through all the preprocessed files to stem the words inside it #
# preprocessed/ --- directory where all the preprocessed emails are       #
# located 																  #
###########################################################################
def stemEmails():
	fileList = []
	fileDir = "preprocessed/"
	for filename in os.listdir(fileDir):
		fileList.append(os.path.join(fileDir, filename))
	for filePath in fileList:
		newList = []
		with open(filePath, "r") as textFile:
			for line in textFile:
				words = word_tokenize(line)
				for word in words:
					newList.append(ps.stem(word))
		with open(filePath, "w") as outFile:
			for i in range(len(newList)):
				outFile.write(newList[i] + " ")

##################################################################
# Traverse through all the words in the dictionary and stem them #
##################################################################
def stemDictionary():
	newList  = []
	with open("dictionary.txt", "r") as textFile:
		for line in textFile:
			words = word_tokenize(line)
			for word in words:
				newList.append(ps.stem(word))
	with open("stemDictionary.txt", "w") as outFile:
		for i in range(len(newList)):
			outFile.write(newList[i])
			outFile.write("\n")

################################################
# Remove common stop words from the dictionary #
################################################
def removeStopWords():
	filteredWords = []
	with open("dictionary.txt", "r") as dictionaryText:
		for line in dictionaryText:
			if line.rstrip() not in stopWords:
				filteredWords.append(line.rstrip())

	with open("removedStopWordsDictionary.txt", "w") as filteredText:
		for word in filteredWords:
			filteredText.write(word)
			filteredText.write("\n")

#####################################################################
# Create a CSV file for easier access of the labels of the test set #
#####################################################################
def getLabelsTest():
	testList = []
	testDir = "test/"
	for testName in os.listdir(testDir):
		testList.append(os.path.join(testDir, testName))
	count = 0
	labelsMatrix = np.zeros(len(testList))
	labelsDir = "testLabels.txt"
	with open(labelsDir, "r") as labels:
		for count, line in enumerate(labels):
			words = line.split()
			check = words[0]
			email = words[1].split("../data/")
			email = email[1]
			if check == 'ham':
				labelsMatrix[count] = 1
			else:
				labelsMatrix[count] = 0

	csvFile = open("dataset-test-labels.csv", "w", newline='')
	writer = csv.writer(csvFile)
	writer.writerow(labelsMatrix.astype(int))
		
	return labelsMatrix

############################################################
# Create a CSV file of the feature vector for the test set #
############################################################
def getFeaturesMatrixTest(dictionary):
	testList = []
	testDir = "test/"
	for testName in os.listdir(testDir):
		testList.append(os.path.join(testDir, testName))
	featuresMatrix = np.zeros((len(testList), len(dictionary)))
	for testIndex, testFile in enumerate(testList):
		with open(testFile, "r") as textTest:
			#print(trainIndex, trainFile)
			for line in textTest:
				#print(trainIndex, line)
				words = line.split()
				#print(words)
				for wordIndex, word in enumerate(dictionary):
					#print(wordIndex, word)
					featuresMatrix[testIndex, wordIndex] = words.count(word)

	csvFile = open("dataset-test.csv", "w", newline='')
	writer = csv.writer(csvFile)
	for index in range(len(testList)):
		writer.writerow(featuresMatrix[index].astype(int))

	return featuresMatrix

######################################################################
# Create a CSV file for easier access of the labels of the train set #
######################################################################
def getLabels():
	trainList = []
	trainDir = "train/"
	for trainName in os.listdir(trainDir):
		trainList.append(os.path.join(trainDir, trainName))
	count = 0
	labelsMatrix = np.zeros(len(trainList))
	labelsDir = "trainLabels.txt"
	with open(labelsDir, "r") as labels:
		for count, line in enumerate(labels):
			#print(line)
			words = line.split()
			check = words[0]
			email = words[1].split("../data/")
			email = email[1]
			#print(check)
			if check == 'ham':
				#print("check: ", check)
				labelsMatrix[count] = 1
			else:
				labelsMatrix[count] = 0
	#print(labelsMatrix)
	#print(len(trainList))

	csvFile = open("dataset-training-labels.csv", "w", newline='')
	writer = csv.writer(csvFile)
	writer.writerow(labelsMatrix.astype(int))

	return labelsMatrix

#############################################################
# Create a CSV file of the feature vector for the train set #
#############################################################
def getFeaturesMatrix(dictionary):
	#print(len(dictionary))
	trainList = []
	trainDir = "train/"
	for trainName in os.listdir(trainDir):
		trainList.append(os.path.join(trainDir, trainName))
	featuresMatrix = np.zeros((len(trainList), len(dictionary)))
	for trainIndex, trainFile in enumerate(trainList):
		with open(trainFile, "r") as textTrain:
			#print(trainIndex, trainFile)
			for line in textTrain:
				#print(trainIndex, line)
				words = line.split()
				#print(words)
				for wordIndex, word in enumerate(dictionary):
					#print(wordIndex, word)
					featuresMatrix[trainIndex, wordIndex] = words.count(word)
					
	#print(featuresMatrix)
	csvFile = open("dataset-training.csv", "w", newline='')
	writer = csv.writer(csvFile)
	for index in range(len(trainList)):
		#listMatrix = list(featuresMatrix[index].astype(int))
		#print(listMatrix)
		#writer.writerow(listMatrix[index])
		writer.writerow(featuresMatrix[index].astype(int))

	#print(featuresMatrix.shape)
	#new = np.array_split(featuresMatrix, 2)
	#print(new[0].shape)
	return featuresMatrix

###################################################################################
# Preprocess the original email files by removing the header, html tags, symbols, #
# and non-English words 														  #
###################################################################################
def getMail(filename):
	text = " "
	with open(filename, "r", errors="ignore") as mailFile:
		for line in mailFile:
			if line == "\n":
				for line in mailFile:
					text += line

	message = ""
	soup = BeautifulSoup(text, "html.parser")
	message = soup.get_text()
	#print(message)
	message = message.lower()
	#print(message)
	message = message.translate(str.maketrans("","", string.digits))
	#print(message)
	message = message.translate(str.maketrans("","", string.punctuation))
	#print(message)
	plain = " ".join(w for w in nltk.wordpunct_tokenize(message) if w.lower() in words or w.isalpha())
	#print(plain)

	#outFile = open("output.txt", "w")
	#outFile.write(plain)
	outname = ((filename.split("."))[1])
	outputname = "outmail.{}".format(outname)
	outFile = open(outputname,"w", errors="ignore")
	outFile.write(plain)

def main():
	#################################
	# Possible values for i: 		#
	# 1: getMail() : preprocessing	#
	# 2: generate dictionary.txt 	#
	# 3: getFeaturesMatrix()		#
	# 	 getFeaturesMatrixTest()	#
	#	 getLabels() 				#
	# 	 getLabelsTest() 			#
	#	: feature vectors & labels 	#
	# 4: Naive Bayes Models 		#
	# 5: removeStopWords()   		#
	# 6: stemDictionary() 			#
	# 7: stemEmails() 				#
	#################################
	i = 4
	if i == 1:
		fileList = []
		fileDir = "data/"
		for filename in os.listdir(fileDir):
			fileList.append(os.path.join(fileDir, filename))
		#print(fileList)
		for filePath in fileList:
			inputname = filePath
			getMail(inputname)
	elif i == 2:
		fileList = []
		fileDir = "preprocessed/"
		for filename in os.listdir(fileDir):
			fileList.append(os.path.join(fileDir, filename))
		with open("dictionary.txt", "w") as outFile:
			for filePath in fileList:
				with open(filePath, "r") as textFile:
					for line in textFile:
						data = line.split(" ")
					for i in data:
						outFile.write(i)
						outFile.write("\n")
		textList = []
		newList = []
		with open("dictionary.txt", "r") as textFile:
			for line in textFile:
				textList.append(line)
		uWords = set(textList)
		for i in range(len(textList)):
			if textList[i].isalpha():
				newList.append(textList[i])
		with open("dictionary.txt", "w") as outFile:
			for i in range(len(newList)):
				outFile.write(newList[i])
				outFile.write("\n")
		with open("dictionary.txt", "r") as textFile:
			lines = textFile.readLines()
			lines.sort()
		with open("dictionary.txt", "w") as outFile:
			for i in lines:
				outFile.write(i)
	elif i == 3:
		dictionary = []
		with open("dictionary.txt", "r") as dicionaryText:
			for line in dicionaryText:
				word = line.split()
				dictionary += word

		trainFeatures = getFeaturesMatrix(dictionary)
		testFeatures = getFeaturesMatrixTest(dictionary)
		trainLabels = getLabels()
		testLabels = getLabelsTest()
	elif i == 4:
		aClass = np.array([0,1])

		datasetTrain = np.genfromtxt("dataset-training.csv", dtype=np.int, delimiter=",",)
		datasetTest = np.genfromtxt("dataset-test.csv", dtype=np.int, delimiter=",")
		datasetTrainLabels = np.genfromtxt("dataset-training-labels.csv", dtype=np.int, delimiter=",")
		datasetTestLabels = np.genfromtxt("dataset-test-labels.csv", dtype=np.int, delimiter=",")

		modelM = MultinomialNB(alpha=1)
		modelB = BernoulliNB(alpha=1)

		splitTrain = datasetTrain.shape[0]//2
		newTrainFeatures0, newTrainFeatures1 = datasetTrain[:splitTrain,:], datasetTrain[splitTrain:,:]

		newTrainLabels0 = datasetTrainLabels[0:splitTrain]
		newTrainLabels1 = datasetTrainLabels[splitTrain:]

		modelM.partial_fit(newTrainFeatures0, newTrainLabels0, classes=aClass)
		modelM.partial_fit(newTrainFeatures1, newTrainLabels1, classes=aClass)

		modelB.partial_fit(newTrainFeatures0, newTrainLabels0, classes=aClass)
		modelB.partial_fit(newTrainFeatures1, newTrainLabels1, classes=aClass)

		#modelM.fit(datasetTrain, datasetTrainLabels)
		#modelB.fit(datasetTrain, datasetTrainLabels)

		predictionsM = modelM.predict(datasetTest)
		predictionsM1 = modelM.predict(datasetTrain)

		predictionsB = modelB.predict(datasetTest)
		predictionsB1 = modelB.predict(datasetTrain)

		modelAccuracyM = accuracy_score(datasetTestLabels, predictionsM)
		modelAccuracyM1 = accuracy_score(datasetTrainLabels, predictionsM1)

		modelAccuracyB = accuracy_score(datasetTestLabels, predictionsB)
		modelAccuracyB1 = accuracy_score(datasetTrainLabels, predictionsB1)

		print("Multinomial Naive Bayes (Test):\t", modelAccuracyM)
		print("Multinomial Naive Bayes (Train):\t", modelAccuracyM1)

		print("Bernoulli Naive Bayes (Test):\t", modelAccuracyB)
		print("Bernoulli Naive Bayes (Train):\t", modelAccuracyB1)
	elif i == 5:
		removeStopWords()
	elif i == 6:
		stemDictionary()
	elif i == 7:
		stemEmails()
		
if __name__ == "__main__":
	main()