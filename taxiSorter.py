import time
import cv2 as cv
import numpy as np
import pytesseract as pt
from pytesseract import Output
from glob import glob
import imutils
import matplotlib.pylab as plt

from collections import deque
import easyocr

#H: 30+ seems good
#S: 210-250 starts with H at 30
#V: 


lower_yellow_bound = np.array([0,138,140])
upper_yellow_bound = np.array([55,220,190])



pt.pytesseract.tesseractCmd = "/usr/local/Cellar/tesseract/5.2.0"


images = glob('/Users/zakmineiko/Documents/imageProcessing/midway/*.JPG')

def findHSV(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h,s,v = 0,220,190
    MAXHUE = 179
    low=np.array([0,138,140])
    up=np.array([h,s,v]) 
    i=10
    while i < 179:
        up=np.array([i,v,s]) 
        mask = cv.inRange(hsv, low, up)
        print("showing V value for " + str(h) + ", "+ str(s) + ", " + str(i))
        plt.imshow(mask)    
        #plt.show()
        plt.pause(0.01)
        plt.draw()
        i+=10
    
    mask = cv.inRange(hsv, lower_yellow_bound, upper_yellow_bound)


def spruceImage(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower_yellow_bound, upper_yellow_bound)
    contours = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    center = None

    if len(contours)>0:
        c = max(contours, key= cv.contourArea)
        ((x,y), radius) = cv.minEnclosingCircle(c)
        M = cv.moments(c)
        center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
        x1 = int(M["m10"]/M["m00"])
        y1 = int(M["m01"]/M["m00"])
    crop = img[y1-200:y1+200, x1-200:x1+200]
    
    #plt.imshow(crop)
    #plt.show()
    return crop

def noCrop(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower_yellow_bound, upper_yellow_bound)
    reader = easyocr.Reader(['en'])
    res = reader.readtext(mask, detail=0)
    return res


def imageToText(img):
    reader = easyocr.Reader(['en'])
    easyRes = reader.readtext(img, detail=0)
    ptRes = pt.image_to_data(img, output_type=Output.STRING)

    return easyRes, ptRes


def loop():    
    taxi_numbers = []
    i=0
    for car in images:
        img = cv.imread(images[i])
        spruced_image = spruceImage(img)
        easyResText, ptResText = imageToText( spruced_image)
        try:
            taxi_numbers.append(easyResText[1])
            taxi_numbers.append(ptResText)
        except:
            print("no number found, using noCrop")
            t = noCrop(img)
            taxi_numbers.append("noCar: " +  str(t) )
        #taxi_numbers.append(text[1])
        #print(text[1])
        i+=1
    print(taxi_numbers)


loop()

#image = cv.imread(images[0])
#findHSV(image)
#spruced_image = spruceImage(image)
#text = imageToText( spruced_image)
#print(text[1])
