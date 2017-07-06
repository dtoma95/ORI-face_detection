This is a simple neural network designed to classify pictures into two classes: ones that contain a face and ones that don't.

* Project contents:
	- train.py - trains the neural network and saves the resulting weights as "end_result.h5" (there is no need to run this script)
	- validate.py - runs predictions for each image in the data/validation directory and prints out the results (these images have to be 32x32)
	- detectImage.py - requests the path of an image and tries to detect images in it (be careful not to add images that are too large)
	- googleSearch.py - requests a google search query, the first result from that google search will be run through the image detection process

* Examples of the training images used can be found in data/train. A total of 73233 32x32 pixel pictures were used to train the neural netowrk:
	- 13213 pictures of faces from http://vis-www.cs.umass.edu/lfw/ (the images form this dataset were cropped and resized to  fit the 32x32 pixel criteria)
	- 59980 pictures of random objects (32x32) from https://www.cs.toronto.edu/~kriz/cifar.html

* Requirements:
	- python 2.7
	- kears and openCV
    - for googleSearch.py to work you have to have Firefox installed and geckodriver.exe in the same directory as googleSearch.py (https://github.com/mozilla/geckodriver/releases)
    - for googleSearch.py to work you have to install selenium ("pip install selenium" should work)
	- for train.py you have to have sklearn installed
	
