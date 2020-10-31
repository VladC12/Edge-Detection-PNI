import cv2
import matplotlib.pyplot as plt
import numpy as np
import Operators as op #in the same directory created by us

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
# plt.imshow(InputGray,cmap = 'gray')
# plt.suptitle('The gray scale input image')
# plt.show()

#Image matrix dimensions
ImageLengthX=InputGray.shape[0] #horizontal length of image
ImageLengthY=InputGray.shape[1] #vertical length of image

(thresh, InputBW) = cv2.threshold(InputGray, 127, 255, cv2.THRESH_BINARY) #converting to black and white
MatrixBW = np.array(InputBW)
#display BW picture
#plt.figure()
#plt.imshow(InputBW,cmap = 'gray')
#plt.show()

print('Choose 1 for Sobel or 2 for Prewitt Operator:')
oper = int(input())
print('What size?')
size = int(input())

maskx = np.zeros((size,size), dtype=int)
masky = np.zeros((size,size), dtype=int)


if oper == 1:
    ##Applying masks
    #Apply a mask in X & Y
    maskx = op.Sobel(size)[0]
    masky = op.Sobel(size)[1]
elif oper == 2:
    ##Applying masks
    #Apply a mask in X & Y
    maskx = op.Prewitt(size)[0]
    masky = op.Prewitt(size)[1]
  
EdgeDetecX = np.zeros((ImageLengthX,ImageLengthY), dtype=int)

InputMini = np.zeros((size,size), dtype=int)

i=0 
j=0
for i in range(ImageLengthX-1):
    for j in range(ImageLengthY-1):
        
        #TODO: Redo InputMini for new algorithm
        
        EDVal = np.sum(np.multiply(InputMini,maskx))
        if EDVal < 0:
            EdgeDetecX[i,j] = 255
        elif EDVal >= 0:
            EdgeDetecX[i,j] = 0

Gx = np.array(EdgeDetecX)
# plt.figure()
# plt.imshow(Gx,cmap = 'gray')
# plt.show()

EdgeDetecY = np.zeros((ImageLengthX,ImageLengthY), dtype=int)

i=0 
j=0
for i in range(ImageLengthX-1):
    for j in range(ImageLengthY-1):
         
        #TODO: Redo InputMini for new algorithm
        
        EDVal = np.sum(np.multiply(InputMini,masky))
        if EDVal < 0:
            EdgeDetecY[i,j] = 255
        elif EDVal >= 0:
            EdgeDetecY[i,j] = 0
            
Gy = np.array(EdgeDetecY)            
# plt.figure()
# plt.imshow(Gy,cmap = 'gray')
# plt.show()

#Normalize the results of both masks
EdgeDetec = np.hypot(Gx,Gy)
plt.figure()
plt.imshow(EdgeDetec,cmap = 'gray')
plt.show()