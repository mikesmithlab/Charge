from particletracker import track_gui
from labvision.images import gaussian_blur, threshold, read_img, display, find_contours, sort_contours, draw_polygon, bgr_to_gray, crop, BBox, draw_contours,find_connected_components
import cv2
import numpy as np

pathname='C:\\Users\\ppzmis\\OneDrive - The University of Nottingham\\Documents\\Papers\\Charge\\ChargeProjectProcessedData\\torsional_spring_constant\\'
filename = 'tag.png'


img = read_img(pathname + filename)
gray_img = bgr_to_gray(img)
blur_img = gaussian_blur(gray_img, kernel=(15,15))
th_img = threshold(blur_img, 90)
#display(th_img)
contour = sort_contours(find_contours(th_img))[-1]
x, y, w, h = cv2.boundingRect(contour)


#vertices=np.array([[x,y],[x,y+h],[x+w,y+h],[x+w,y]])

crop_img = crop(th_img, BBox(x-6,x+w+6, y-6,y+h+6))
components = find_connected_components(crop_img, connectivity=4, option=cv2.CV_32S)

print('labels')
print(components[0])
print('stats')
print(components[1])

#display(crop_img)
print(np.shape(crop_img))
indices = np.argwhere(components[0]==3)
print(np.shape(indices))
new_img = np.zeros(np.shape(crop_img))

inertia = 0
num_pixels = components[1][3][4]
print('area')
print(num_pixels)
mass_tag = 0.0302E-3#kg
width_tag = 7.8E-3#m
scale=width_tag/w #m/px
mass_per_px = mass_tag/num_pixels

print(scale)
middle = int((w+12)/2)

inertia_tag=0
for index in indices:
    dx2=(scale*(index[1]-middle) )**2
    inertia_tag = inertia_tag+mass_per_px*dx2

print(inertia_tag)

inertia_bead=1.73E-9

inertia=inertia_bead+inertia_tag
print(inertia)   






