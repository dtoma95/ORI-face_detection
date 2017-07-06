# Ultimate folder of stuff\SCHOOL\G3S2\ORI\SW21-2014\ORI-face_detection
# Tomislav Dobricki SW21/2014 FTN - Novi Sad

# import the necessary packages
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense
import numpy as np
import cv2
import os
import math
from keras.callbacks import ModelCheckpoint

# define the architecture of the network
model = Sequential()
model.add(Dense(768, input_dim=3072, kernel_initializer="uniform",
	activation="relu"))
model.add(Dense(384, kernel_initializer="uniform", activation="relu"))
model.add(Dense(2))
model.add(Activation("softmax"))

print("[INFO] loading weights...")
model.load_weights('end_result.h5')


def resize_to_normal(image):
	currentHeight, currentWidth, channels = image.shape
	area = 256*256
	print currentHeight, currentWidth
	if currentHeight*currentWidth<= area:
		return image
	
	print "Resizing the picture to a managable size..."
	aspectRatio =   float(currentWidth)/ currentHeight
	
	height = math.sqrt(area / aspectRatio)
	width = height * aspectRatio
	return cv2.resize(image, (int(width), int(height)))

def image_to_32flat(image, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	#cv2.imshow("cropped", cv2.resize(image, size))
	#cv2.waitKey(0)
	
	return cv2.resize(image, size).flatten()

data = []
labels = []

input = raw_input("Enter the image path: ")

image = cv2.imread(input)
#cv2.imwrite('grey.jpg',image)

image = resize_to_normal(image)

max_height, max_width, channels = image.shape


step = 6


if max_height > max_width:
	dim = max_width
else:
	dim = max_height

while dim >= 100:
	#i and j represent the coordinates of the top left corner of the image we are cropping
	i = 0
	j = 0
	while i <= max_height-dim:
		
		while j <= max_width-dim:
			crop_img = image[i:i+dim, j:j+dim]
			features = image_to_32flat(crop_img)
			#print features.size
			data.append(features)
			
			labels.append((i,j,dim))
			j += 3
		
		i += 3
		j = 0
	dim = dim - 32
	
	
	

	
data = np.array(data) / 255.0
lista = model.predict(data,batch_size=10,verbose=0)

found = False
j = 0
for i in lista:
	found = True
	if i[0] > 0.8:
		crop_img = image[labels[j][0]:labels[j][0]+labels[j][2], labels[j][1]:labels[j][1]+labels[j][2]]
		#cv2.imshow("cropped", crop_img)
		#cv2.waitKey(0)
		cv2.rectangle(image, (labels[j][1], labels[j][0]), (labels[j][1]+labels[j][2], labels[j][0]+labels[j][2]), (255,0,0), 1)

	j+=1
if(found == False):
	print "No faces were detected in this image"
else:
	print "Faces were detected in this image"
cv2.imshow("Face detection", image)
cv2.waitKey(0)
