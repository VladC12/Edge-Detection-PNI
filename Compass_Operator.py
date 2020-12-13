#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 16:08:11 2020

@author: vlad
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import Operators as op #in the same directory created by us

def rotate_ring(matrix, offset):
    dim = len(matrix[0])
    #print(dim)
    last_element = matrix[offset][offset]
    for j in range(1 + offset, dim - offset):
        matrix[offset][j-1] = matrix[offset][j]
    matrix[offset][dim-1-offset] = matrix[1+offset][dim-1-offset]
    for i in range(1 + offset, dim - offset):
        matrix[i-1][dim-1-offset] = matrix[i][dim-1-offset]
    matrix[dim-1-offset][dim-1-offset] = matrix[dim-1-offset][dim-2-offset]
    for j in range(1+offset, dim-offset):
        matrix[dim-1-offset][dim-j] = matrix[dim-1-offset][dim-j-1]
    matrix[dim-1-offset][offset] = matrix[dim-2-offset][offset]
    for i in range(1+offset, dim-offset):
        matrix[dim-i][offset] = matrix[dim-i-1][offset]
    matrix[1+offset][offset] = last_element
    return matrix


def rotate_matrix(matrix):
    dim = len(matrix[0])
    for offset in range(0, int(dim/2)):
        matrix = rotate_ring(matrix, offset)
    return matrix
    

def compass_filter(gray, kernel):
    #result  = np.zeros(gray.shape, dtype=np.float32)
    results = []
    results.append(np.array([kernel])) #append the first value 
    for i in range(7):
        kernel_ = np.array([rotate_matrix(kernel)]) #rotate the kernel
        results.append(kernel_) #appending to results
     
    k1 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, results[0][0]), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    k2 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, results[1][0]), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    k3 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, results[2][0]), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    k4 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, results[3][0]), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    k5 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, results[4][0]), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    k6 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, results[5][0]), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    k7 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, results[6][0]), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    k8 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, results[7][0]), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    
    magn = cv2.max(
        k1, cv2.max(
            k2, cv2.max(
                k3, cv2.max(
                    k4, cv2.max(
                        k5, cv2.max(
                            k6, cv2.max(
                                k7, k8
                            )
                        )
                    )
                )
            )
        )
    )
    return magn

 
size = 2
while size % 2 == 0: #insert the size of the kernel
    print('What size? (must be odd)')
    size = int(input())
    
kernel = op.Sobel(size)[0]
print(kernel)
fg = cv2.imread("E:/Facultate/2020-2021_Sem1_An_IV/PNI/AnacondaStuff/_TestImagesPNI_labs/ImgTstColor/Lenna.bmp")
fg = cv2.cvtColor(fg, cv2.COLOR_BGR2RGB)

plt.figure()
plt.imshow(fg)
plt.suptitle('The input color image') #not needed just for demonstration purposes
plt.show()

fg = cv2.imread("E:/Facultate/2020-2021_Sem1_An_IV/PNI/AnacondaStuff/_TestImagesPNI_labs/ImgTstColor/Lenna.bmp",0)
#display gray
plt.imshow(fg,cmap = 'gray')
plt.suptitle('The gray scale input image')
plt.show()

fg_rgb = cv2.cvtColor(fg, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(fg_rgb, cv2.COLOR_RGB2GRAY)    
output = compass_filter(gray, kernel)
plt.imshow(output, interpolation='none', cmap='gray')
plt.xticks([]), plt.yticks([])
plt.show()
