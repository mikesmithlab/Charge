from labvision.video import ReadVideo
from labvision.images.basics import display, write_img, read_img
from labvision.images.thresholding import threshold, absolute_diff
from labvision.images.morphological import opening
from labvision.images import bgr_to_gray, gaussian_blur, find_contours, sort_contours, draw_polygon, find_circles, draw_circle, find_connected_components, find_contours, sort_contours, rotated_bounding_rectangle, draw_contours, cut_out_object

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_scale(img, width=9.8E-3):
    """Work out image scale"""
    coords =display(img)
    width_pixels = np.abs(coords[1][0] - coords[0][0])
    scale = width / width_pixels
    height_px = int(coords[0][1])
    print('Scale is : {}'.format(scale))
    print('Width_pixels : {}'.format(width_pixels))
    return scale, width_pixels, height_px
    

if __name__=='__main__':
    path='E:/RawData/Mike/2023_05_12/'
    file1='P1001656.mp4'
    file2='P1001657.mp4'

    vid=ReadVideo(path + file1)
    img = vid.read_frame(n=1072)
    pts=display(img)
    scale, _,_=get_scale(img)
    vid=ReadVideo(path + file2)
    img2 = vid.read_frame(n=vid.num_frames-1)
    pts2=display(img2)
    #get_scale(img)

    print(pts)
    print(pts2)

    theta = (pts[0][0]-pts2[0][0])*scale

    mg=0.568e-3*9.81

    F=mg*theta
    print(F)
