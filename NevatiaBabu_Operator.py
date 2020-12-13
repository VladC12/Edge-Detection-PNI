#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 15:52:32 2020

@author: vlad
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2

#this is compass operator implemented with sobel kernels (3X3)

def nevatia_babu(gray):
    kernel1 = np.array([[-100, -100, 0, 100, 100],
                        [-100, -100, 0, 100, 100],
                        [-100, -100, 0, 100, 100],
                        [-100, -100, 0, 100, 100],
                        [-100, -100, 0, 100, 100]], dtype = np.float32) #0-deg orientation
    
    kernel2 = np.array([[-100, 32, 100, 100, 100],
                        [-100, 78, 92, 100, 100],
                        [-100, -100, 0, 100, 100],
                        [-100, -100, -92, 78, 100],
                        [-100, -100, -100, -32, 100]], dtype = np.float32) #30-deg orientation
    
    kernel3 = np.array([[100, 100, 100, 100, 100],
                        [-32, 78, 100, 100, 100],
                        [-100, -92, 0, 92, 100],
                        [-100, -100, -100, -78, 32],
                        [-100, -100, -100, -100, -100]], dtype = np.float32) #60-deg orientation
    
    kernel4 = np.array([[100, 100, 100, 100, 100],
                        [100, 100, 100, 100, 100],
                        [0, 0, 0, 0, 0],
                        [-100, -100, -100, -100, -100],
                        [-100, -100, -100, -100, -100]], dtype = np.float32) #90-deg orientation
    
    kernel5 = np.array([[100, 100, 100, 100, 100],
                        [100, 100, 100, 78, -32],
                        [100, 92, 0, -92, -100],
                        [32, -78, -100, -100, -100],
                        [-100, -100, -100, -100, -100]], dtype = np.float32) #120-deg orientation
    
    kernel6 = np.array([[100, 100, 100, 32, -100],
                        [100, 100, 92, -78, -100],
                        [100, 100, 0, -100, -100],
                        [100, 78, -92, -100, -100],
                        [100, -32, -100, -100, -100]], dtype = np.float32) #150-deg orientation
    
  
    #CV_32F - the pixel can have any value between 0-1.0
    #NORM_MINMAX
    #CV_8UC1 - number of channels
    # https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#normalize
    
    k1 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernel1), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    k2 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernel2), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    k3 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernel3), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    k4 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernel4), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    k5 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernel5), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    k6 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernel6), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
  

    magn = cv2.max(
        k1, cv2.max(
            k2, cv2.max(
                k3, cv2.max(
                    k4, cv2.max(
                        k5, k6
                    )
                )
            )
        )
    )
    return magn

def Nevatia(input_file, size, kernel):
    fg = cv2.imread(input_file)
    fg = cv2.cvtColor(fg, cv2.COLOR_BGR2RGB)
    
    plt.figure()
    plt.imshow(fg)
    plt.suptitle('The input color image') #not needed just for demonstration purposes
    plt.show()

    fg = cv2.imread(input_file,0)
    #display gray
    plt.imshow(fg,cmap = 'gray')
    plt.suptitle('The gray scale input image')
    plt.show()

    fg_rgb = cv2.cvtColor(fg, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(fg_rgb, cv2.COLOR_RGB2GRAY)    
    output = nevatia_babu(gray)
    plt.imshow(output, interpolation='none', cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.show()
    
    return output
