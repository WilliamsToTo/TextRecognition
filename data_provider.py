import cv2 as cv
import numpy as np
import os
from PIL import Image

def provide_oneImage(path):
    #print(path)
    img = cv.imread(path, 0)
    #print(type(img))
    img_flat = img.flatten()
    
    return img_flat

def provide_data(path):
    #path = '../train_data_resized/'
    files = np.array(['0','4',  '8',  '_b',  '_d' , '_f' , '_h',  '_j',  '_l',  '_n',  '_p',  '_r',  '_t',  '_v',  '_x',  '_z',
    '1', '5', '9', 'B', 'D', 'F', 'H',   'J',   'L',   'N',   'P',   'R',   'T',   'V',   'X' ,  'Z',
    '2', '6', '_a', '_c', '_e', '_g',  '_i',  '_k',  '_m' , '_o',  '_q',  '_s',  '_u',  '_w',  '_y',
    '3', '7', 'A', 'C', 'E', 'G', 'I',   'K',   'M' ,  'O' ,  'Q' ,  'S' ,  'U',  'W' , 'Y'
    ])
    
    files2 = np.array(['0','4','8','_b'])
    
    #train_data = np.zeros((1,784))
    #train_labels = np.zeros((1,1))
    train_data = []
    train_labels = []
    
    index = 0
    for file in files:
        #print(type(path),type(file))
        #print(file)
        #directory = os.fsencode(path+str(file))
        directory = path+str(file)+'/'
        print(file)
        
        for image in os.listdir(directory):
            #imagename = os.fsencode(image)
            #print(image)
            imagePath = str(directory)+str(image)
            one_example = provide_oneImage(imagePath)
            
            #train_data = np.vstack((train_data, one_example))
            #train_labels = np.vstack((train_labels, file))
            train_data.append(one_example)
            train_labels.append(index)
        index += 1
    
    return (np.array(train_data, dtype='f'), np.array(train_labels))

#(data, labels) = provide_data('../test_data_resized/')
        
        
        