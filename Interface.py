import EdgeDetection as ed
import Compass_Operator as co
import NevatiaBabu_Operator as nbo
import Operators as op

import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os

#input_file="E:/Facultate/2020-2021_Sem1_An_IV/PNI/AnacondaStuff/_TestImagesPNI_labs/ImgTstColor/Lenna.bmp"

Operator = [
    'Sobel',
    'Sobel w/ Angle',
    'Prewitt',
    'Prewitt w/ Angle',
    'Compass',
    'Nevatia Babu'
    ]

root = tk.Tk()
root.title("Edge Detection by: Crehul Vlad & Hornai Vlad")

variable = tk.StringVar(root)
variable.set(Operator[0])

e = tk.Entry(root, width = 20)
e.grid(row = 0, column = 3, columnspan=3)

opt = tk.OptionMenu(root, variable, *Operator)
opt.grid(row=0, column=0, columnspan=3)

def choose():
    global input_file
    input_file = filedialog.askopenfilename(initialdir = os.getcwd(), title = "Select image file", filetype=(("BMP file", "*.bmp"),("JPG file", "*.jpg"),("PNG file", "*.png"),("All files", "*.*")))
    
    og = ImageTk.PhotoImage(Image.open(input_file))
    original.image = og
    original.config(image = original.image)
    
input_file =""

forsize = cv2.imread(input_file)

btn = tk.Button(root, text = "Browse image", command = choose)
btn.grid(row=2,column=1,columnspan=3)

def ok():
    if variable.get() == 'Sobel w/ Angle':
        oper = 1
        if (int(e.get()) % 2) != 0:
            size = int(e.get())
        else: size = 3
        Gy = ed.EdgeDetection(input_file, oper, size)[0]
        Gx = ed.EdgeDetection(input_file, oper, size)[1]
        G = ed.EdgeDetectionNormalized(Gy, Gx)
        Edge = ed.EdgeDetectionAngle(Gy, Gx, G)
    elif variable.get() == 'Sobel':
        oper = 1
        if (int(e.get()) % 2) != 0:
            size = int(e.get())
        else: size = 3
        Gy = ed.EdgeDetection(input_file, oper, size)[0]
        Gx = ed.EdgeDetection(input_file, oper, size)[1]
        Edge = ed.EdgeDetectionNormalized(Gy, Gx)
    elif variable.get() == 'Prewitt w/ Angle':
        oper = 2
        if (int(e.get()) % 2) != 0:
            size = int(e.get())
        else: size = 3
        Gy = ed.EdgeDetection(input_file, oper, size)[0]
        Gx = ed.EdgeDetection(input_file, oper, size)[1]
        G = ed.EdgeDetectionNormalized(Gy, Gx)
        Edge = ed.EdgeDetectionAngle(Gy, Gx, G)
    elif variable.get() == 'Prewitt':
        oper = 2
        if (int(e.get()) % 2) != 0:
            size = int(e.get())
        else: size = 3
        Gy = ed.EdgeDetection(input_file, oper, size)[0]
        Gx = ed.EdgeDetection(input_file, oper, size)[1]
        Edge = ed.EdgeDetectionNormalized(Gy, Gx)
    elif variable.get() == 'Compass':
        if (int(e.get()) % 2) != 0:
            size = int(e.get())
        else: size = 3
        Edge = co.Compass(input_file, size, op.Sobel(size)[0])
    elif variable.get() == 'Nevatia Babu':
        if (int(e.get()) % 2) != 0:
            size = int(e.get())
        else: size = 3
        Edge = nbo.Nevatia(input_file, size, op.Sobel(size)[0]) 
    
    img = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(Edge)))
    
    #result = tk.Label(root, image = img)
    result.image = img
    result.config(image = result.image)
      
button = tk.Button(root, text='Compile', command=ok).grid(row=3,column=1,columnspan=3)

original = tk.Label(root)
original.grid(row=1,column=0,columnspan=3)

result = tk.Label(root)
result.grid(row=1,column=3,columnspan=3)
  
root.mainloop()