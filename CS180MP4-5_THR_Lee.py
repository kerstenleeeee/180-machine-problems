import cv2
import sys
import numpy as np 
import os
import time 
from PIL import Image
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

def train():
	##########################
	# get train set (1 to 6) #
	##########################
	print("Trainining...")
	imageList = []
	imageDir = "att_faces/"
	for (subDir, mainDir, files) in os.walk(imageDir):
		for subDir in mainDir:
			#print(subDir)
			filepath = os.path.join(imageDir, subDir)
			for filename in os.listdir(filepath):
				#print(filename)
				words = filename.split(".")
				#print(words[0])
				if int(words[0]) < 7:
					#print(words[0])
					imgPath = filepath + "/" + filename
					#print(imgPath)
					imageList.append(imgPath)
	images = []
	labels = []
	label = 0
	count = 0
	for img in imageList:
		if count == 6:
			count = 0
			label = label + 1
		pixels = cv2.imread(img, 0)
		images.append(pixels)
		labels.append(label)
		count = count + 1

	images = np.asarray(images)
	images = images.reshape(len(images), -1)
	labels = np.asarray(labels)
	images = normalize(images, norm="l2")


	#####################
	# start of training #
	#####################

	start = time.time()

	#############
	# SVM model #
	#############

	'''modelSVM = SVC()
	modelSVM.fit(images, labels)'''

	#############
	# ANN model #
	#############

	modelANN = MLPClassifier(hidden_layer_sizes=(5152,))
	modelANN.fit(images, labels)

	#############
	# PCA model #
	#############

	'''modelPCA = PCA(n_components=10)
	imgPCA = modelPCA.fit_transform(images, labels)
	modelANN.fit(imgPCA, labels)'''

	end = time.time()
	print("Training Complete")

	print("Training Time: ", end - start)
	print("Predicting Train Set...")

	#################
	# Model predict #
	#################

	#modelSVM.predict(images)
	modelANN.predict(images)
	#modelANN.predict(imgPCA)

	print("Prediction of Train Set Complete")
	#print(modelSVM.score(images, labels))
	print(modelANN.score(images, labels))
	#print(modelANN.score(imgPCA, labels))

	#test(modelSVM)
	test(modelANN)
	#test(modelANN, modelPCA)

def test(modelANN):
#def test(modelSVM):
#def test(modelANN, modelPCA):
	################
	# get test set #
	################
	print("Predicting Test Set...")
	imageList = []
	imageDir = "att_faces/"
	for (subDir, mainDir, files) in os.walk(imageDir):
		for subDir in mainDir:
			filepath = os.path.join(imageDir, subDir)
			for filename in os.listdir(filepath):
				words = filename.split(".")
				if int(words[0]) > 6:
					imgPath = filepath + "/" + filename
					imageList.append(imgPath)
	images = []
	labels = []
	label = 0
	count = 0
	for img in imageList:
		if count == 4:
			count = 0
			label = label + 1
		pixels = cv2.imread(img, 0)
		images.append(pixels)
		labels.append(label)
		count = count + 1

	images = np.asarray(images)
	images = images.reshape(len(images), -1)
	labels = np.asarray(labels)
	images = normalize(images, norm="l2")

	##############################
	# PCA fitting & transforming #
	##############################

	#imgPCA = modelPCA.transform(images, labels)

	####################
	# Model prediction #
	####################
	
	#modelSVM.predict(images)	
	modelANN.predict(images)
	#modelANN.predict(imgPCA)

	print("Prediction of Test Set Complete")
	#print(modelSVM.score(images, labels))
	print(modelANN.score(images, labels))
	#print(modelANN.score(imgPCA, labels))

def main():
	train()
	#test()


if __name__ == "__main__":
	main()