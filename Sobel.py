import cv2
import matplotlib.pyplot as plt
import numpy as np

###Edge Detection Algotirhm (Sobel Operator)

InputFile="E:/Facultate/2020-2021_Sem1_An_IV/PNI/AnacondaStuff/_TestImagesPNI_labs/ImgTstColor/Lenna.bmp" 
#file name here 
#TODO: change the constant file location to a command line argument and then in the future argument for program
InputRGB = cv2.imread(InputFile)


#Convert the image to black and white
InputGray=cv2.imread(InputFile,0) #converting to gray
#display rgb
plt.figure()
plt.imshow(InputRGB)
plt.suptitle('The input color image') #not needed just for demonstration purposes
plt.show()
#display gray
plt.imshow(InputGray,cmap = 'gray')
plt.suptitle('The gray scale input image')
plt.show()

#Image matrix dimensions
ImageLengthX=InputGray.shape[0] #horizontal length of image
ImageLengthY=InputGray.shape[1] #vertical length of image

(thresh, InputBW) = cv2.threshold(InputGray, 127, 255, cv2.THRESH_BINARY) #converting to black and white

plt.figure()
plt.imshow(InputBW,cmap = 'gray')
plt.show()

##Applying masks
#Apply a mask in X
maskx = np.array([[-1, 0, 1],  #3x3 Sobel Operator
                  [-2, 0, 2],  #TODO: Upgrade to 5x5, 7x7, etc.
                  [-1, 0, 1]])

SobelX = np.zeros((ImageLengthX,ImageLengthY), dtype=int)

i=0 
j=0
for i in range(ImageLengthX-1):
    for j in range(ImageLengthY-1):
        InputMini = np.array([[InputBW.item(i-1, j-1),InputBW.item(i-1, j),InputBW.item(i-1, j+1)],
                              [InputBW.item(i,   j-1),InputBW.item(i,   j),InputBW.item(i,   j+1)],
                              [InputBW.item(i+1, j-1),InputBW.item(i+1, j),InputBW.item(i+1, j+1)]])
        #edge (literally) case
        if i-1 < 0:
            InputMini[0,:] = 0
        if j-1 < 0:
            InputMini[:,0] = 0
        if i+1 > ImageLengthX:
            InputMini[2,:] = 0
        if j+1 >ImageLengthY:
            InputMini[:,2] = 0
        
        SobVal = np.sum(np.multiply(InputMini,maskx))
        if SobVal < 0:
            SobelX[i,j] = 255
        elif SobVal >= 0:
            SobelX[i,j] = 0

Gx = np.array(SobelX)
plt.figure()
plt.imshow(Gx,cmap = 'gray')
plt.show()

#Apply a mask in Y
masky = np.array([[-1, -2, -1], #3x3 Sobel Operator 
                  [ 0,  0,  0], #TODO: Upgrade to 5x5, 7x7, etc.
                  [ 1,  2,  1]])

SobelY = np.zeros((ImageLengthX,ImageLengthY), dtype=int)

i=0 
j=0
for i in range(ImageLengthX-1):
    for j in range(ImageLengthY-1):
        InputMini = np.array([[InputBW.item(i-1, j-1),InputBW.item(i-1, j),InputBW.item(i-1, j+1)],
                                 [InputBW.item(i,   j-1),InputBW.item(i,   j),InputBW.item(i,   j+1)],
                                 [InputBW.item(i+1, j-1),InputBW.item(i+1, j),InputBW.item(i+1, j+1)]])
        #edge (literally) case
        if i-1 < 0:
            InputMini[0,:] = 0
        if j-1 < 0:
            InputMini[:,0] = 0
        if i+1 > ImageLengthX:
            InputMini[2,:] = 0
        if j+1 >ImageLengthY:
            InputMini[:,2] = 0
        
        SobVal = np.sum(np.multiply(InputMini,masky))
        if SobVal < 0:
            SobelY[i,j] = 255
        elif SobVal >= 0:
            SobelY[i,j] = 0
            
Gy = np.array(SobelY)            
plt.figure()
plt.imshow(Gy,cmap = 'gray')
plt.show()

#Normalize the results of both masks
Sobel = np.hypot(Gx,Gy)
plt.figure()
plt.imshow(Sobel,cmap = 'gray')
plt.show()