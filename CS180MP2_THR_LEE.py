########################################
# Segmentation using KMeans Clustering #
# Random centroids                     #
# Manually selected centroids          #
# RGB, HSV, CIELa*b* colorspaces       #
#                                      #
# LEE, Kristine-Clair                  #
# CS180                                #
########################################

import sys
import cv2
import numpy as np
from scipy.misc.pilutil import imread
from scipy.misc.pilutil import imsave
from sklearn.cluster import KMeans
import os, os.path

#####################################################
# converting to the different color spaces function #
#####################################################
def convertColorSpace(img):
	im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	#im = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	#im = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
	return im

####################
# random centroids #
####################
def train(img1, img2, k, x):
	rows1 = img1.shape[0]
	cols1 = img1.shape[1]
	channel1 = img1.shape[2]

	rows2 = img2.shape[0]
	cols2 = img2.shape[1]
	channel2 = img2.shape[2]

	image1 = img1.reshape((rows1*cols1), channel1)
	image2 = img2.reshape((rows2*cols2), channel2)

	newImage = np.concatenate((image1, image2), axis =0)

	km = KMeans(n_clusters = k)
	km.fit(newImage)
	clusters = np.array(km.cluster_centers_)
	labels = np.array(km.labels_)
	labels = labels.reshape(rows1*2, cols1)
	#cv2.imwrite("out.jpg", labels)
	#labels.append(img1.shape[2])
	imsave("out.jpg", labels)
	#imsave("D:/Owl/CS180/MP2/output/out.jpg", labels)

	'''im = imread("filaria.jpg")
	im = convertColorSpace(im)
	rows0 = im.shape[0]
	cols0 = im.shape[1]
	channel0 = im.shape[2]
	image = im.reshape(rows0*cols0, channel0)
	new = km.predict(image)
	labels0 = np.array(new)
	labels0 = labels0.reshape(rows0, cols0)
	#labels0 = cv2.cvtColor(labels0, cv2.COLOR_HSV2BGR)
	imsave("outerer.jpg", labels0)'''
	#labels0 = cv2.cvtColor(im, cv2.COLOR_HSV2BGR)
	##############################################
	if x == 1:
		i = 1
		imageList = []
		imageDir = "filarioidea/"
		#imageDir = "plasmodium/"
		#imageDir = "schistoma/"
		imageExte = ".jpg"
		for filename in os.listdir(imageDir):
			extension = os.path.splitext(filename)[1]
			if extension.lower() != imageExte:
				continue
			imageList.append(os.path.join(imageDir, filename))
		for imagePath in imageList:
			#image = cv2.imread(imagePath)
			#cluster(image, centroids, 3, i)
			im = imread(imagePath)
			im = convertColorSpace(im)
			rows1 = im.shape[0]
			cols1 = im.shape[1]
			channel1 = im.shape[2]
			image = im.reshape(rows1*cols1, channel1)
			new = km.predict(image)
			labels1 = np.array(new)
			labels1 = labels1.reshape(rows1, cols1)
			output = "filaria_clustered{}.jpg".format(i)
			imsave(output, labels1)
			i = i + 1
	elif x == 2:
		i = 1
		imageList = []
		imageDir = "schistoma/"
		#imageDir = "plasmodium/"
		#imageDir = "schistoma/"
		imageExte = ".jpg"
		for filename in os.listdir(imageDir):
			extension = os.path.splitext(filename)[1]
			if extension.lower() != imageExte:
				continue
			imageList.append(os.path.join(imageDir, filename))
		for imagePath in imageList:
			#image = cv2.imread(imagePath)
			#cluster(image, centroids, 3, i)
			im = imread(imagePath)
			im = convertColorSpace(im)
			rows1 = im.shape[0]
			cols1 = im.shape[1]
			channel1 = im.shape[2]
			image = im.reshape(rows1*cols1, channel1)
			new = km.predict(image)
			labels1 = np.array(new)
			labels1 = labels1.reshape(rows1, cols1)
			output = "schistoma_clustered{}.jpg".format(i)
			imsave(output, labels1)
			i = i + 1
	elif x == 3:
		i = 1
		imageList = []
		imageDir = "plasmodium/"
		#imageDir = "plasmodium/"
		#imageDir = "schistoma/"
		imageExte = ".jpg"
		for filename in os.listdir(imageDir):
			extension = os.path.splitext(filename)[1]
			if extension.lower() != imageExte:
				continue
			imageList.append(os.path.join(imageDir, filename))
		for imagePath in imageList:
			#image = cv2.imread(imagePath)
			#cluster(image, centroids, 3, i)
			im = imread(imagePath)
			im = convertColorSpace(im)
			rows1 = im.shape[0]
			cols1 = im.shape[1]
			channel1 = im.shape[2]
			image = im.reshape(rows1*cols1, channel1)
			new = km.predict(image)
			labels1 = np.array(new)
			labels1 = labels1.reshape(rows1, cols1)
			output = "plasmodium_clustered{}.jpg".format(i)
			imsave(output, labels1)
			i = i + 1

###############################
# manually selected centroids #
###############################
def manual(img1, img2, k, x):
	rows1 = img1.shape[0]
	cols1 = img1.shape[1]
	channel1 = img1.shape[2]

	rows2 = img2.shape[0]
	cols2 = img2.shape[1]
	channel2 = img2.shape[2]

	image1 = img1.reshape((rows1*cols1), channel1)
	image2 = img2.reshape((rows2*cols2), channel2)

	#print(img1)
	pixel1 = img1[1, 2]
	pixel2 = img2[3, 4]	
	pixel3 = img2[5, 6]

	newImage = np.concatenate((image1, image2), axis =0)

	pixels = []
	pixels.append(pixel1)
	pixels.append(pixel2)
	pixels.append(pixel3)
	#print(np.array(pixels))

	km = KMeans(n_clusters = k, init = np.array(pixels), n_init = 1)
	km.fit(newImage)
	clusters = np.array(km.cluster_centers_)
	labels = np.array(km.labels_)
	labels = labels.reshape(rows1*2, cols1)
	#cv2.imwrite("out.jpg", labels)
	#print(clusters)
	imsave("out.jpg", labels)

	'''im = imread("105c.jpg")
	im = convertColorSpace(im)
	rows0 = im.shape[0]
	cols0 = im.shape[1]
	channel0 = im.shape[2]
	image = im.reshape(rows0*cols0, channel0)
	new = km.predict(image)
	labels0 = np.array(new)
	labels0 = labels0.reshape(rows0, cols0)
	imsave("outer.jpg", labels0)'''

	####################################################
	if x == 1:
		i = 1
		imageList = []
		imageDir = "filarioidea/"
		#imageDir = "plasmodium/"
		#imageDir = "schistoma/"
		imageExte = ".jpg"
		for filename in os.listdir(imageDir):
			extension = os.path.splitext(filename)[1]
			if extension.lower() != imageExte:
				continue
			imageList.append(os.path.join(imageDir, filename))
		for imagePath in imageList:
			#image = cv2.imread(imagePath)
			#cluster(image, centroids, 3, i)
			im = imread(imagePath)
			im = convertColorSpace(im)
			rows1 = im.shape[0]
			cols1 = im.shape[1]
			channel1 = im.shape[2]
			image = im.reshape(rows1*cols1, channel1)
			new = km.predict(image)
			labels1 = np.array(new)
			labels1 = labels1.reshape(rows1, cols1)
			output = "filaria_clustered{}.jpg".format(i)
			imsave(output, labels1)
			i = i + 1
	elif x == 2:
		i = 1
		imageList = []
		imageDir = "schistoma/"
		#imageDir = "plasmodium/"
		#imageDir = "schistoma/"
		imageExte = ".jpg"
		for filename in os.listdir(imageDir):
			extension = os.path.splitext(filename)[1]
			if extension.lower() != imageExte:
				continue
			imageList.append(os.path.join(imageDir, filename))
		for imagePath in imageList:
			#image = cv2.imread(imagePath)
			#cluster(image, centroids, 3, i)
			im = imread(imagePath)
			im = convertColorSpace(im)
			rows1 = im.shape[0]
			cols1 = im.shape[1]
			channel1 = im.shape[2]
			image = im.reshape(rows1*cols1, channel1)
			new = km.predict(image)
			labels1 = np.array(new)
			labels1 = labels1.reshape(rows1, cols1)
			output = "schistoma_clustered{}.jpg".format(i)
			imsave(output, labels1)
			i = i + 1
	elif x == 3:
		i = 1
		imageList = []
		imageDir = "plasmodium/"
		#imageDir = "plasmodium/"
		#imageDir = "schistoma/"
		imageExte = ".jpg"
		for filename in os.listdir(imageDir):
			extension = os.path.splitext(filename)[1]
			if extension.lower() != imageExte:
				continue
			imageList.append(os.path.join(imageDir, filename))
		for imagePath in imageList:
			#image = cv2.imread(imagePath)
			#cluster(image, centroids, 3, i)
			im = imread(imagePath)
			im = convertColorSpace(im)
			rows1 = im.shape[0]
			cols1 = im.shape[1]
			channel1 = im.shape[2]
			image = im.reshape(rows1*cols1, channel1)
			new = km.predict(image)
			labels1 = np.array(new)
			labels1 = labels1.reshape(rows1, cols1)
			output = "plasmodium_clustered{}.jpg".format(i)
			imsave(output, labels1)
			i = i + 1

def main():
	what = input("Random(1) or Manual(2)\n")
	if what == '1':
		k = 3
		k1 = 2
		#print("filaria")
		img1 = imread("filaria.jpg")
		img2 = imread("filaria4.jpg")
		#img1 = convertColorSpace(img1)
		#img2 = convertColorSpace(img2)
		#imsave("fudge.jpg", img1)
		#imsave("fudge1.jpg",img2)
		train(img1, img2, k, 1)
		
		#print("schistoma")
		img1 = imread("schistoma5.jpg")
		img2 = imread("schistoma6.jpg")
		img1 = convertColorSpace(img1)
		img2 = convertColorSpace(img2)
		train(img1, img2, k, 2)

		#print("plasmodium")
		img1 = imread("7c.jpg")
		img2 = imread("55c.jpg")
		img1 = convertColorSpace(img1)
		img2 = convertColorSpace(img2)
		train(img1, img2, k1, 3)
	elif what == '2':
		img1 = imread("filaria.jpg")
		img2 = imread("filaria4.jpg")
		img1 = convertColorSpace(img1)
		img2 = convertColorSpace(img2)
		manual(img1, img2, 3, 1)
		
		img1 = imread("schistoma5.jpg")
		img2 = imread("schistoma6.jpg")
		img1 = convertColorSpace(img1)
		img2 = convertColorSpace(img2)
		manual(img1, img2, 3, 2)

		img1 = imread("7c.jpg")
		img2 = imread("55c.jpg")
		img1 = convertColorSpace(img1)
		img2 = convertColorSpace(img2)
		manual(img1, img2, 2, 3)


	'''img1 = imread("filaria.jpg")
	img2 = imread("filaria4.jpg")
	#img1 = imread("schistoma5.jpg")
	#img2 = imread("schistoma6.jpg")
	#img1 = imread("7c.jpg")
	#img2 = imread("55c.jpg")
	img1 = convertColorSpace(img1)
	img2 = convertColorSpace(img2)

	
	#print(what)

	if what == '1':
		k = 3
		choice = 1
		train(img1, img2, k, choice)
	elif what == '2':
		k = 2
		choice = 0
		#print("OK")
		manual(img1, img2, k, choice)'''

###############
# initializer #
###############
if __name__ == "__main__":
	main()