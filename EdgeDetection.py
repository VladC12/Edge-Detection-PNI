import cv2
import matplotlib.pyplot as plt
import numpy as np
import Operators as op #in the same directory created by us

### Edge Detection Algotirhm (Sobel Operator)

#InputFile="E:/Facultate/2020-2021_Sem1_An_IV/PNI/AnacondaStuff/_TestImagesPNI_labs/ImgTstColor/Lenna.bmp" 
# file name here 
# TODO: change the constant file location to a command line argument and then in the future argument for program

def EdgeDetection(InputFile, oper, size):
    InputRGB = cv2.imread(InputFile)


    ## Convert the image to black and white
    InputGray=cv2.imread(InputFile,0) #converting to gray
    ## display rgb
    InputRGB = cv2.cvtColor(InputRGB, cv2.COLOR_BGR2RGB)


    # Image matrix dimensions
    ImageLengthX=InputGray.shape[0] #horizontal length of image
    ImageLengthY=InputGray.shape[1] #vertical length of image

    (thresh, InputBW) = cv2.threshold(InputGray, 127, 255, cv2.THRESH_BINARY) #converting to black and white
    MatrixBW = np.array(InputBW)

    maskx = np.zeros((size,size), dtype=int)
    masky = np.zeros((size,size), dtype=int)


    ## Applying masks
    if oper == 1:
        #Apply a mask in X & Y
        maskx = op.Sobel(size)[0]
        masky = op.Sobel(size)[1]
    elif oper == 2:
        #Apply a mask in X & Y
        maskx = op.Prewitt(size)[0]
        masky = op.Prewitt(size)[1]
  
    EdgeDetecX = np.zeros((ImageLengthX,ImageLengthY), dtype=int)
    
    InputMini = np.zeros((size,size), dtype=int)

    paddim = int(size/2)
    Padded = np.pad(MatrixBW, ((paddim, paddim),(paddim, paddim)), 'constant')

    i=0
    j=0
    for i in range(ImageLengthX):
        for j in range(ImageLengthY):
        
            InputMini = Padded[i:i+size,j:j+size]
    
            EDVal = np.sum(np.multiply(InputMini,maskx))
            if EDVal < 0:
                EdgeDetecX[i,j] = 255
            elif EDVal >= 0:
                EdgeDetecX[i,j] = 0

    Gx = np.array(EdgeDetecX)

    EdgeDetecY = np.zeros((ImageLengthX,ImageLengthY), dtype=int)

    i=0
    j=0
    for i in range(ImageLengthX):
        for j in range(ImageLengthY):
            
            InputMini = Padded[i:i+size,j:j+size]
                            
            EDVal = np.sum(np.multiply(InputMini,masky))
            if EDVal < 0:
                EdgeDetecY[i,j] = 255
            elif EDVal >= 0:
                EdgeDetecY[i,j] = 0
            
    Gy = np.array(EdgeDetecY)            
    
    return Gy, Gx

def EdgeDetectionNormalized(Gy, Gx):
    # Normalize the results of both masks
    EdgeDetec = np.hypot(Gx,Gy)
    
    return EdgeDetec

def EdgeDetectionAngle(Gy, Gx, EdgeDetec):
    # Defining Colors
    red = np.array([255, 0, 0])
    blue = np.array([0, 0, 255])
    green = np.array([0, 255, 0])

    ## orientation
    orien_map = np.arctan2(Gy, Gx) * 180 / np.pi
    
    # rgb orientation
    EdgeDetecOrien = np.zeros((orien_map.shape[0], orien_map.shape[1], 3), dtype=int)
    i=0
    j=0
    for i in range(Gx.shape[1]):
        for j in range(Gy.shape[0]):
            if (orien_map[i,j] < 90) and (EdgeDetec[i,j] == 255):
                EdgeDetecOrien[i,j] = red
            elif (orien_map[i,j] >= 90) and (EdgeDetec[i,j] == 255):
                EdgeDetecOrien[i,j] = blue
            elif (EdgeDetec[i,j] > 255):
                EdgeDetecOrien[i,j] = green

    
    return EdgeDetecOrien