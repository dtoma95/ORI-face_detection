# Ultimate folder of stuff\SCHOOL\G3S2\ORI\SW21-2014\ORI-face_detection
import os
import json
import urllib2
import sys
import time
import argparse
import cv2
import math
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense
from keras.utils import np_utils
from imutils import paths

# adding path to geckodriver to the OS environment variable
# assuming that it is stored at the same path as this script
os.environ["PATH"] += os.pathsep + os.getcwd()
download_path = "googleResults/"

def googleSearch(searchtext, result_index):
	
	url = "https://www.google.co.in/search?q="+searchtext+"&source=lnms&tbm=isch"
	driver = webdriver.Firefox()
	driver.get(url)

	headers = {}
	headers['User-Agent'] = "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36"
	extensions = {"jpg", "jpeg", "png"}
	
	


	images = driver.find_elements_by_xpath("//div[@class='rg_meta notranslate']")
	if(len(images) == 0):
		print "No images found for query: \""+ searchtext +"\" "
		return
	img = images[result_index]
	
		
	img_url = json.loads(img.get_attribute('innerHTML'))["ou"]
	img_type = json.loads(img.get_attribute('innerHTML'))["ity"]
	print "Downloading image from: ", img_url
	try:
		if img_type not in extensions:
			img_type = "jpg"
		req = urllib2.Request(img_url, headers=headers)
		raw_img = urllib2.urlopen(req).read()
		f = open(download_path+searchtext.replace(" ", "_")+"."+img_type, "wb")
		f.write(raw_img)
		f.close
	except Exception as e:
		print "Download failed:", e
	
	print "Downloaded result no."+ str(result_index+1) + " image for query: \""+ searchtext +"\" "
	driver.quit()
	return download_path+searchtext.replace(" ", "_")+"."+img_type


def resize_to_normal(image):
	currentHeight, currentWidth, channels = image.shape
	area = 256*256
	print currentHeight, currentWidth
	if currentHeight*currentWidth<= area:
		return image
	
	print "Resizing the downloaded picture to a managable size..."
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
	
def detect_face(path):
	data = []
	labels = []

	image = cv2.imread(path)
	
	image = resize_to_normal(image)
	
	max_height, max_width, channels = image.shape

	if max_height > max_width:
		dim = max_width
	else:
		dim = max_height

	while dim >= 64:
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
		if(dim != 64):
			dim = dim - 32
			if(dim < 64):
				dim = 64
		else:
			dim = dim-32
	
	data = np.array(data) / 255.0
	lista = model.predict(data,batch_size=10,verbose=0)

	j = 0
	found = False
	for i in lista:
		if i[0] > 0.8:
			found = True
			crop_img = image[labels[j][0]:labels[j][0]+labels[j][2], labels[j][1]:labels[j][1]+labels[j][2]]
		
			cv2.rectangle(image, (labels[j][1], labels[j][0]), (labels[j][1]+labels[j][2], labels[j][0]+labels[j][2]), (255,0,0), 1)

		j+=1
	if(found == False):
		print "No faces were detected in this image"
	else:
		print "Faces were detected in this image"
	cv2.imshow("Face detection", image)
	cv2.waitKey(0)
	
	
if __name__ == "__main__":
	# Initialize neural network
	model = Sequential()
	model.add(Dense(768, input_dim=3072, kernel_initializer="uniform",
		activation="relu"))
	model.add(Dense(384, kernel_initializer="uniform", activation="relu"))
	model.add(Dense(2))
	model.add(Activation("softmax"))

	print("[INFO] loading weights...")
	model.load_weights('end_result.h5')
	
	while True:
		input = raw_input("Enter the desired search query: ")
		if input == "":
			break;
		inputs = input.split(",")
		query = inputs[0]
		if(len(inputs) == 2):
			result_index = int(inputs[1]) -1
		else:
			result_index = 0
		path = googleSearch(query, result_index)
		detect_face(path)
	