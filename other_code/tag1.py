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

path='Z:/GranularCharge/BlueBeads7mm/2023_03_24/'

for i in range(1,1000):

    plt.close('all')
    if i <10:
        fname=path+'img1__00'+str(i)+'.png'
    elif i < 100:
        fname=path+'img1__0'+str(i)+'.png'
    else:
        fname=path+'img1__'+str(i)+'.png'
        
    if os.path.exists(fname) == False:
        break
    
    image = cv2.imread(fname)
    output = image.copy()
    #img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img=image[:,:,0]
            

            
    slice=img[300][:]
    #plt.figure(3)
    #plt.plot(slice)
    tag=slice>100
    len=np.sum(tag)
    lsave=np.append(lsave,len)         
          
    k=np.arange(1264)
    xc=np.sum(k*tag)/len
    
    isave=np.append(isave,i)
    xsave=np.append(xsave,xc)

    print(i,xc,len)

t=isave*2/60
dx=(xsave-xsave[0])*3.9/115
#np.savetxt('drift.txt',np.array([t,dy]).T)

plt.figure(1)
plt.plot(t,dx,'b*')
plt.xlabel('t (hours)')
plt.ylabel('dx (mm)')

plt.figure(2)
plt.plot(t,lsave,'g*')
plt.xlabel('t (hours)')
plt.ylabel('len (pix)')

#plt.figure(2)
#plt.plot(isave*2/60,-(xsave-xsave[0])*3.9/115,'b*')
#plt.xlabel('t (hours)')
#plt.ylabel('dx (mm)')

plt.show()


