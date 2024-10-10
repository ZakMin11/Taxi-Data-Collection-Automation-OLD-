import pytesseract as pt
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import time
from glob import glob
import cv2


car_files = glob('/Users/zakmineiko/Documents/imageProcessing/images/*')

license_cascade = cv2.CascadeClassifier('/Users/zakmineiko/Documents/imageProcessing/license_plates.xml')
car_cascade = cv2.CascadeClassifier('/Users/zakmineiko/Documents/imageProcessing/cars.xml')


def plt_show(image, title="", gray = False, size=(1000,1000)):
    temp = image
    if gray == False:
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        plt.title(title)
        plt.imshow(temp, cmap='gray')
        plt.show()

def detect_car(img):
    temp = img
    gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    car = car_cascade.detectMultiScale(gray, scaleFactor=None, minNeighbors=10, minSize=(300,300))
    print("number cars detected: "+str(len(car)))
    for cars in car:
        (x,y,w,h) = cars
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+h]
        cv2.rectangle(temp, (x,y), (x+w, y+h), (0,255,0), 10)

def detect_number(img):
    temp = img
    gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    number = license_cascade.detectMultiScale(gray,2,1,100)
    print("number plate detected: "+str(len(number)))
    for numbers in number:
        (x,y,w,h) = numbers
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+h]
        cv2.rectangle(temp, (x,y), (x+w, y+h), (0,255,0), 10)
    plt_show(temp)


def show_all_images():
    i = 0
    
    for car in car_files:
        img = cv2.imread(car_files[i])
        detect_number(img)
        plt_show(img)
        i+=1
        print(i)


def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes	
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")
    
#show_all_images()
img = cv2.imread(car_files[3])
detect_car(img)
plt_show(img)
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#bfilter = cv2.bilateralFilter(gray, 30,17,17)
#edged = cv2.Canny(bfilter,30,200)
#plt_show(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))

#detect_number(img)
#plt_show(img)
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2_imshow(gray)

#img_matPlotLib = plt.imread(car_files[1]) #3D numpy array
#img_cv2 = cv2.imread(car_files[1]) # also numpy array, width/height swapped

#print(img_matPlotLib.shape)
#print (type(img_cv2))

#---plot and show image
#fig, ax = plt.subplots(figsize = (10,10))
#ax.imshow(img_matPlotLib)
#plt.show()

#---img_resized = cv2.resize(img, None, fx = 0.25, fy = 0.25) #resize image 
#img_cv2RGB = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB) # convert BGR to RGB
#img_resized = cv2.resize(img_cv2RGB, None, fx = 0.25, fy = 0.25)

#fig, ax = plt.subplots(figsize=(10,10))
#ax.imshow(img_cv2RGB)
#ax.axis('off')
#plt.show()

#cars = face_cascade.detectMultiScale(img_resized, 1.1, 2)
#for (x,y,w,h) in cars:
#    cv2.rectangle(img_resized, (x,y),(x+w,y+h),(0,0,255),2)

#def detect_number(img):
#    car = img.copy()
#    car_rect = face_cascade.detectMultiScale(car, scaleFactor=1.2,minNeighbors=5)
#    for (x,y,w,h) in car_rect:
#        ((x + w, y + h), (255, 255, 255), 10)
#    return car

#carIm = detect_car(img_resized)



#cv2.imshow("Result", detect_car(img_resized))
#for car in car_files:
#    grey = cv2.cvtColor(car, cv2.COLOR_RGB2GRAY)
    