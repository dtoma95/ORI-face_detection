
# Tomislav Dobricki SW21/2014 FTN - Novi Sad


# import the necessary packages

from keras.models import Sequential
from keras.layers import Activation

from keras.layers import Dense
from keras.utils import np_utils
from imutils import paths
import numpy as np
import argparse
import cv2
import os
from keras.callbacks import ModelCheckpoint

def image_to_feature_vector(image, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()

# define the architecture of the network
model = Sequential()
model.add(Dense(768, input_dim=3072, kernel_initializer="uniform",
	activation="relu"))
model.add(Dense(384, kernel_initializer="uniform", activation="relu"))
model.add(Dense(2))
model.add(Activation("softmax"))

print("[INFO] loading weights...")
model.load_weights('end_result.h5')

	
	
data = []
labels = []
imagePaths = list(paths.list_images("data\\validation\\faces_validation"))
for (i, imagePath) in enumerate(imagePaths):
		# load the image and extract the class label (assuming that our
		# path as the format: /path/to/dataset/{class}.{image_num}.jpg	
	
	image = cv2.imread(imagePath)

		# construct a feature vector raw pixel intensities, then update
		# the data matrix and labels list
	features = image_to_feature_vector(image)
	print features.size
	data.append(features)
	labels.append(os.path.basename(imagePath))
		# show an update every 10 images
	if i > 0 and i % 10 == 0:
		print("[INFO] processed {}/{}".format(i, len(imagePaths)))
		
data = np.array(data) / 255.0
lista = model.predict(data,batch_size=10,verbose=0)
print "[Validation] Input of 20 pictures of faces"
j = 0
for i in lista:
	if i[0] < i[1]:
		print i
		print "NOT_FACE" + " - " + labels[j]
	else:
		print "FACE" + " - " + labels[j]
	j+=1

	
data = []
labels = []
imagePaths = list(paths.list_images("data\\validation\\not_faces_validation"))
for (i, imagePath) in enumerate(imagePaths):
		# load the image and extract the class label (assuming that our
		# path as the format: /path/to/dataset/{class}.{image_num}.jpg	
	
	image = cv2.imread(imagePath)

		# construct a feature vector raw pixel intensities, then update
		# the data matrix and labels list
	features = image_to_feature_vector(image)
	data.append(features)
	labels.append(os.path.basename(imagePath))
		# show an update every 10 images
	if i > 0 and i % 10 == 0:
		print("[INFO] processed {}/{}".format(i, len(imagePaths)))
		

data = np.array(data) / 255.0
lista = model.predict(data,batch_size=10,verbose=0)
print "[Validation] Input of 21 pictures without faces"
j = 0
for i in lista:
	if i[0] < i[1]:
		print "NOT_FACE" + " - " + labels[j]
	else:
		print "FACE" + " - " + labels[j]
	j+=1
		
		
data = []
labels = []
imagePaths = list(paths.list_images("data\\validation\\wild_faces_validation"))
for (i, imagePath) in enumerate(imagePaths):
		# load the image and extract the class label (assuming that our
		# path as the format: /path/to/dataset/{class}.{image_num}.jpg	
	
	image = cv2.imread(imagePath)

		# construct a feature vector raw pixel intensities, then update
		# the data matrix and labels list
	features = image_to_feature_vector(image)
	data.append(features)
	labels.append(os.path.basename(imagePath))
		# show an update every 10 images
	if i > 0 and i % 10 == 0:
		print("[INFO] processed {}/{}".format(i, len(imagePaths)))
		

data = np.array(data) / 255.0
lista = model.predict(data,batch_size=10,verbose=0)
print "[Validation] Test pictures gathered from random sources"
j = 0
for i in lista:
	if i[0] < i[1]:
		print "NOT_FACE" + " - " + labels[j]
	else:
		print "FACE" + " - " + labels[j]
	j+=1