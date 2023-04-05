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

path='Z:/GranularCharge/BlueBeads7mm/2023_03_23/'
filestub='img3__'

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


        #plt.imshow(image)
        # Find circles
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 
                                1.3, 100,
                                param1=160, #120
                                param2=17,  #17
                                minRadius=112,
                                maxRadius=117)

        # If some circle is found
        if circles is not None:
            #print(i)
            # Get the (x, y, r) as integers
            circles = np.round(circles[0, :]).astype("int")
            #print(i,np.shape(circles)[0],circles[0][2])
            #print(circles[0][0])
            # loop over the circles
            for (x, y, r) in circles:
                cv2.circle(output, (x, y), r, (255, 255, 255), 2)
                # show the output image
                
                if show:
                    cv2.namedWindow('Circle', cv2.WINDOW_KEEPRATIO)
                    cv2.imshow('Circle', output)

                    #print(img[437][:])
                    #plt.plot(circles[0][0],circles[0][1],'b*')
                    cv2.resizeWindow('Circle', 632,508)
                    cv2.waitKey(1000)
                #print('Got here')
                #print(np.shape(img))
                xc=circles[0][0]
                yc=circles[0][1]
                ra=circles[0][2]
                print(i,xc,yc,ra)
                
                isave=np.append(isave,i)
                ysave=np.append(ysave,yc)
                xsave=np.append(xsave,xc)

                slice=img[yc-300][:]
                #plt.figure(3)
                #plt.plot(slice)
                len=np.sum(slice>100)
                lsave=np.append(lsave,len)         
                
        else:
            print(i,'none')
            
    t=isave*2/60
    dy=(ysave[0]-ysave)*scale
    #np.savetxt('drift.txt',np.array([t,dy]).T)

    plt.figure(1)
    plt.plot(t,dy,'b*')
    plt.xlabel('t (hours)')
    plt.ylabel('dy (mm)')

    plt.figure(2)
    plt.plot(t,lsave,'g*')
    plt.xlabel('t (hours)')
    plt.ylabel('len (pix)')

    

    #plt.figure(2)
    #plt.plot(isave*2/60,-(xsave-xsave[0])*3.9/115,'b*')
    #plt.xlabel('t (hours)')
    #plt.ylabel('dx (mm)')

    plt.pause(120)
    


