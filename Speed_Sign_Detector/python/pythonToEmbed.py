import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
ray = np.zeros((60,40))
def plotStdVectors(data_in):
    global ray
    print ray
    ray = np.array(data_in)
    ray = ray.reshape((60,40))
    #x = ray[0:-1:2]
    #y = ray[1:-1:2]
    #y = np.append(y,ray[-1])
    #print x.shape
    #print y.shape
    print "Printing from Python in plotStdVectors()"
    
    cv.imshow("Image",ray)
    cv.waitKey(0)
    return 0
