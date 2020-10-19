import cv2
import matplotlib.pyplot as plt
import numpy as np

###Edge Detection Algotirhm (Sobel Operator)

InputFile="E:/Facultate/2020-2021_Sem1_An_IV/PNI/AnacondaStuff/_TestImagesPNI_labs/ImgTstColor/avion_a.jpg" 
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

##Applying masks
#Apply a mask in X
maskx = np.array([[-1, -2, -1], #3x3 Sobel Operator 
                  [ 0,  0,  0], #TODO: Upgrade to 5x5, 7x7, etc.
                  [-1, -2, -1]])

i=0 
j=0
InputGrayMini = np.array([[InputGray.item(i-1, j-1),InputGray.item(i-1, j),InputGray.item(i-1, j+1)],
                          [InputGray.item(i,   j-1),InputGray.item(i,   j),InputGray.item(i,   j+1)],
                          [InputGray.item(i+1, j-1),InputGray.item(i+1, j),InputGray.item(i+1, j+1)]])

if i-1 < 0:
    InputGrayMini[i-1,:] = 0
#if j-1 < 0:
#    InputGrayMini[j-1,:] = 0
#if i+1 > ImageLengthX:
#    InputGrayMini[i+1,:] = 0
#if j+1 >ImageLengthY:
#    InputGrayMini[j+1,:] = 0
    
test = np.multiply(InputGrayMini,maskx)
x = np.sum(np.multiply(InputGrayMini,maskx))

#Apply a mask in Y
masky = np.array([[-1, 0, -1],  #3x3 Sobel Operator
                  [-2, 0, -2],  #TODO: Upgrade to 5x5, 7x7, etc.
                  [-1, 0, -1]])

#Normalize the results of both masks


#Plot the results
