#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 16:52:09 2021

@author: mike
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os.path


plt.close('all')

isave=np.array([])
ysave=np.array([])
xsave=np.array([])
lsave=np.array([])
t=np.array([])

path='Z:/GranularCharge/BlueBeads7mm/2023_03_23/'
filestub='img4__'

bead_diam=7.8
bead_pixels=762-533
scale=bead_diam/bead_pixels
tag_offset = (526-348)

show=True

while True:

    for i in range(1,1000):

        plt.close('all')
        if i <10:
            fname=path+filestub+'00'+str(i)+'.png'
        elif i < 100:
            fname=path+filestub+'0'+str(i)+'.png'
        else:
            fname=path+filestub+str(i)+'.png'
            
        if os.path.exists(fname) == False:
            break
        
        image = cv2.imread(fname)
        output = image.copy()
        #img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img=image[:,:,0]       

        slice=img[300-5:300+5][:]
        #plt.figure(3)
        #plt.plot(slice)
        len=np.sum(slice>100)
        lsave=np.append(lsave,len)  
        t=np.append(t, i*2/60)       
                
    
    
    #np.savetxt('drift.txt',np.array([t,dy]).T)


    plt.figure(2)
    plt.plot(t,lsave,'g*')
    plt.xlabel('t (hours)')
    plt.ylabel('len (pix)')

    

    #plt.figure(2)
    #plt.plot(isave*2/60,-(xsave-xsave[0])*3.9/115,'b*')
    #plt.xlabel('t (hours)')
    #plt.ylabel('dx (mm)')

    plt.pause(120)
    


