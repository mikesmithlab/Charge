from labvision.video import ReadVideo
from labvision.images.basics import display
from labvision.images.thresholding import threshold, adaptive_threshold
from labvision.images import bgr_to_gray, gaussian_blur, find_contours, sort_contours, draw_contours,draw_polygon
from particletracker import track_gui
import cv2
import matplotlib.pyplot as plt
import numpy as np


pathname = 'C:\\Users\\ppzmis\\Documents\\'
filename = 'bluebead_torsion3.avi'

#track_gui(pathname + filename)

readVid = ReadVideo(pathname + filename)

width=[]

for img in readVid:
    gauss_img = gaussian_blur(bgr_to_gray(img),kernel=(3,3))
    th_img=threshold(gauss_img, 73)
    contour = sort_contours(find_contours(th_img, hierarchy=False))[-1]
    x, y, w, h = cv2.boundingRect(contour)
    width.append(w)
    vertices=np.array([[x,y],[x,y+h],[x+w,y+h],[x+w,y]])
    display(draw_polygon(img,vertices))

framerate = 30

t=np.array(list(range(len(width))))/framerate
width=np.array(width)

np.savetxt(pathname+filename[:-4]+'.csv',np.c_[t,width], delimiter=',',encoding='utf-8')


plt.figure()
plt.plot(t,width,'rx')
plt.show()








