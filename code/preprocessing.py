import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import imageio
image_path='/content/drive/MyDrive/Project/sign dataset'
def loadImages(path,label):
  image_files=sorted([os.path.join(path,label,file)
   for file in os.listdir(path+str('/')+label) if file.endswith('.jpg')
  ])
  return image_files
def preprocess_images(data,label):
    count=0
    for image in data:

        #reading image
        img=imageio.imread(image)
        #Converting image to grayscale
        gray_img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        #Converting image to HSV format
        hsv_img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
        #Defining boundary level for skin color in HSV
        skin_color_lower= np.array([0,40,30],np.uint8)
        skin_color_upper= np.array([43,255,255],np.uint8)
        #Producing mask
        skin_mask=cv2.inRange(hsv_img,skin_color_lower,skin_color_upper)
        #Removing Noise from mask
        skin_mask=cv2.medianBlur(skin_mask,5)
        skin_mask=cv2.addWeighted(skin_mask,0.5,skin_mask,0.5,0.0)
        #Applying Morphological operations
        kernel=np.ones((5,5),np.uint8)
        skin_mask=cv2.morphologyEx(skin_mask,cv2.MORPH_CLOSE,kernel)
        #Extracting hand by applying mask
        hand=cv2.bitwise_and(gray_img,gray_img,mask=skin_mask)
        #Get edges by Canny edge detection
        canny=cv2.Canny(hand,60,60)
        #saving preprocessed images
        path='/content/drive/MyDrive/Project/preprocessed dataset'
        final_path = path+"/"+label
        cv2.imwrite(final_path+f'/'+str(count)+'.jpg', canny)
        count+=1
signs=['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
for label in signs:
    images=[]
    images=loadImages(image_path,label)
    preprocess_images(images,label)