####Arbitrary Size Kernels
##For Prewitt and Sobel
#Inspired from: https://www.iasj.net/iasj?func=fulltext&aId=52927 
#and https://stackoverflow.com/questions/9567882/sobel-filter-kernel-of-large-size/41065243#41065243
# https://www.researchgate.net/publication/334001329_Expansion_and_Implementation_of_a_3x3_Sobel_and_Prewitt_Edge_Detection_Filter_to_a_5x5_Dimension_Filter

import numpy as np
import cv2

def Sobel(size):
    SobelOpX = np.zeros((size, size), dtype = int) #initialize a size*size matrix
    SobelOpY = np.zeros((size, size), dtype = int)
    
    p = [(j,i) for j in range(size) 
           for i in range(size) 
           if not (i == (size -1)/2. and j == (size -1)/2.)]
    
    for i,j in p:
            i_ = int(i-(size -1)/2.)
            j_ = int(j-(size -1)/2.)
            SobelOpX[i, j] = (i_ / float(i_*i_ + j_*j_))*size*10
            SobelOpY[i, j] = (j_ / float(i_*i_ + j_*j_))*size*10
    return SobelOpX, SobelOpY #returned as a tuple   
  
def Prewitt(size):
    PrewittOpX = np.zeros((size, size), dtype = int)
    PrewittOpY = np.zeros((size, size), dtype = int)
    
    p = [(j,i) for j in range(size) 
           for i in range(size) 
           if not (i == (size -1)/2. and j == (size -1)/2.)]
    
    for i,j in p:
            i_ = int(i-(size -1)/2.)
            j_ = int(j-(size -1)/2.)
            PrewittOpX[i, j] = i_
            PrewittOpY[i, j] = j_ 
    return PrewittOpX, PrewittOpY

def Convolution(image, kernel, average=False):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    

    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    output = np.zeros(image.shape)
 
    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)
 
    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
 
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]
    return output
'''
print(Convolution(cv2.imread("E:/Facultate/2020-2021_Sem1_An_IV/PNI/AnacondaStuff/_TestImagesPNI_labs/ImgTstColor/Lenna.bmp"), Sobel(3)[0]))
'''