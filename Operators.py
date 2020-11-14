####Arbitrary Size Kernels
##For Prewitt and Sobel
#Inspired from: https://www.iasj.net/iasj?func=fulltext&aId=52927 
#and https://stackoverflow.com/questions/9567882/sobel-filter-kernel-of-large-size/41065243#41065243
# https://www.researchgate.net/publication/334001329_Expansion_and_Implementation_of_a_3x3_Sobel_and_Prewitt_Edge_Detection_Filter_to_a_5x5_Dimension_Filter

import numpy

def Sobel(size):
    SobelOpX = numpy.zeros((size, size), dtype = int) #initialize a size*size matrix
    SobelOpY = numpy.zeros((size, size), dtype = int)
    
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
    PrewittOpX = numpy.zeros((size, size), dtype = int)
    PrewittOpY = numpy.zeros((size, size), dtype = int)
    
    p = [(j,i) for j in range(size) 
           for i in range(size) 
           if not (i == (size -1)/2. and j == (size -1)/2.)]
    
    for i,j in p:
            i_ = int(i-(size -1)/2.)
            j_ = int(j-(size -1)/2.)
            PrewittOpX[i, j] = i_
            PrewittOpY[i, j] = j_ 
    return PrewittOpX, PrewittOpY
