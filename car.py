import time
import cv2
import numpy as np
import pytesseract as pt
from glob import glob
import imutils
from collections import deque
import easyocr

pt.pytesseract.tesseractCmd = "/usr/local/Cellar/tesseract/5.2.0"
#HUE [0,179]
#SATURATION [0,255]
#VALUE [0,255]

#   bgr = np.uint8([[[58,149,170]]])
#    hsvColor = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
#    print(hsvColor)

img = cv2.imread('/Users/zakmineiko/Documents/imageProcessing/midway/car5.JPG')
#img = cv2.resize(img, (744,496))
car_images = glob('/Users/zakmineiko/Documents/imageProcessing/midway/*.JPG')
casc = cv2.CascadeClassifier('/Users/zakmineiko/Documents/imageProcessing/license_plates.xml')
#def filterColor(img):
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


lower_yellow = np.array([0,138,140])
upper_yellow = np.array([54,220,190])



mask = cv2.inRange(hsv, lower_yellow,upper_yellow)

cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
center = None

if len(cnts)>0:
    c = max(cnts, key=cv2.contourArea)
    ((x, y), radius) = cv2.minEnclosingCircle(c)
    M = cv2.moments(c)
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    x1 = int(M["m10"] / M["m00"])
    y1 = int(M["m01"] / M["m00"])

#if radius > 10:
#    cv2.circle(img, center, 5, (0, 0, 255), -1)

crop = img[y1-200:y1+200, x1-200:x1+200]
#print(int(M["m10"]),int(M["m00"]), int(M["m01"]),int(M["m00"]))
#print(int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

result = cv2.bitwise_and(img,img, mask = mask)

plate = casc.detectMultiScale(img,1.2,4)

for (x,y,w,h) in plate:

    #detect plate with rectangle
    #rec. start point (x,y), rec. end point (x+w, y+h), blue color(255,0,0), line width 1

    plates_rec = cv2.rectangle(result, (x,y), (x+w, y+h), (0,255,0), 1)        
    #cv2.putText(plates_rec, 'Text', (x, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    gray_plates = result[y:y+h, x:x+w]
    color_plates = img[y:y+h, x:x+w]

    cv2.imshow('img', gray_plates)
    cv2.waitKey(5000)
    

    height, width, chanel = gray_plates.shape
    #print(height, width)

    #print(pt.image_to_string(result, lang='eng'))
    #print(to.image_to_text(result))
#cv2.imshow('filtered yellow', result)

#cv2.resize(crop, )

custom_config = r'--oem 1 --psm 8 -l eng -c min_characters_to_try=4'
#height, width, channels = crop.shape
#config_str = '--dpi ' + str(height)
#print(pt.image_to_string(crop, config=custom_config))/
#print(pt.image_to_osd(crop, config = ''))


reader = easyocr.Reader(['en'])
res = reader.readtext(crop)
print(res)


cv2.imshow('cropped', crop)
cv2.waitKey(0)


    


#i = 0
#for car in car_images:
#    img = cv2.imread(car_images[i])
#    filterColor(img)
#    print("car ", i)
#    i+=1
#    time.sleep(1)
    
quit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

casc = cv2.CascadeClassifier('/Users/zakmineiko/Documents/imageProcessing/cars2.xml')

carRect = casc.detectMultiScale(gray,1.1,1)

for (x, y, w, h) in carRect:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

filterColor(img)
#cv2.imshow("ya",img)
cv2.waitKey(0)


#edged = cv2.Canny(result, 75, 200)

    #contours = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #contours = imutils.grab_contours(contours)
    #contours = sorted(contours,key=cv2.contourArea, reverse=true)[:10]
    #screenCnt = None
    #for c in contours:
        #pari = cv2.arcLength(c, True)
        #approx = cv2.approxPolyDP(c, 0.018*pari, True)
       # if len(approx)==4:
      #      screenCnt = approx
     #       break

    #if screenCnt is None:
      #  detected = 0
     #   print("No contour detected")
    #else:
    #    detected = 1
   # if detected == 1:
  #      cv2.drawContours(result, [screenCnt], -1, (0,0,255),3)
 #   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
   # mask = np.zeros(gray.shape, np.uint8)
  #  new_img = cv2.drawContours(mask,[screenCnt],0,255,-1)
 #   new_img = cv2.bitwise_and(result, result, mask = mask)
#    cv2.imshow(new_img)    
