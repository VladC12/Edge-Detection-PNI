import cv2
import matplotlib.pyplot as plt
import numpy as np
import Operators as op #in the same directory created by us

### Edge Detection Algotirhm (Sobel Operator)

#InputFile="E:/Facultate/2020-2021_Sem1_An_IV/PNI/AnacondaStuff/_TestImagesPNI_labs/ImgTstColor/Lenna.bmp" 
# file name here 
# TODO: change the constant file location to a command line argument and then in the future argument for program

def EdgeDetection(InputFile, oper, size):
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

    image = cv2.imread(InputFile)

    Gx = op.Convolution(image, maskx)         
    Gy = op.Convolution(image, masky)            
    return Gy, Gx

def EdgeDetectionNormalized(Gy, Gx):
    # Normalize the results of both masks
    Edge = np.hypot(Gx,Gy)
    return Edge * 255.0 / Edge.max()

def EdgeDetectionAngle(Gy, Gx, EdgeDetec):
    
    # Defining Colors
    red = np.array([255, 0, 0])
    blue = np.array([0, 0, 255])
    green = np.array([0, 255, 0])
    yellow = np.array([255, 255, 0])
    
    ## orientation
    orien_map = np.arctan2(Gy , Gx) * 180 / np.pi
    
    # rgb orientation
    EdgeDetecOrien = np.zeros((orien_map.shape[0], orien_map.shape[1], 3), dtype=int)
    i=0
    j=0
    thr = 15
    for i in range(Gx.shape[1]):
        for j in range(Gy.shape[0]):
            if (orien_map[i,j] < 90.0) and (orien_map[i,j] > 0.0) and (EdgeDetec[i,j] > thr):
                EdgeDetecOrien[i,j] = red
            elif (orien_map[i,j] > 90.0) and (orien_map[i,j] < 180.0) and (EdgeDetec[i,j] > thr):
                EdgeDetecOrien[i,j] = blue
            elif (orien_map[i,j] < 0.0) and (orien_map[i,j] > -90.0) and (EdgeDetec[i,j] > thr):
                EdgeDetecOrien[i,j] = green
            elif (orien_map[i,j] < -90.0) and (orien_map[i,j] > -180.0) and (EdgeDetec[i,j] > thr):
                EdgeDetecOrien[i,j] = yellow
    
    return EdgeDetecOrien
'''
Gx = EdgeDetection("E:/Facultate/2020-2021_Sem1_An_IV/PNI/AnacondaStuff/_TestImagesPNI_labs/ImgTstColor/Lenna.bmp", 1, 3)[0]
Gy = EdgeDetection("E:/Facultate/2020-2021_Sem1_An_IV/PNI/AnacondaStuff/_TestImagesPNI_labs/ImgTstColor/Lenna.bmp", 1, 3)[1]
G = EdgeDetectionNormalized(Gx, Gy)

plt.imshow(Gx, cmap='gray')
plt.show()

plt.imshow(Gy, cmap='gray')
plt.show()

plt.imshow(G, cmap='gray')
plt.show()

print("This is Orientation: ", EdgeDetectionAngle(Gy, Gx, G)[1])
print("This is Edge Values: ", EdgeDetectionAngle(Gy, Gx, G)[2])
plt.imshow(EdgeDetectionAngle(Gy, Gx, G)[0])
plt.show()
'''