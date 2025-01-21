import numpy as np
from labvision.images import crop_circle, adaptive_threshold, bgr_to_gray, display, read_img

def find_ball(im):
    bw = adaptive_threshold(im[:,:,2], 20, 1)
    display(bw)
    #return x_pix,y_pix,rad_pix

def find_angle(im):
    x_pix, y_pix, rad_pix = find_ball(im)
    top_left = (y_pix-1, x_pix-1)
    crop_im = im[int(y_pix - rad_pix - 1):int(y_pix + rad_pix + 1),int(x_pix - rad_pix - 1):int(x_pix + rad_pix + 1),:]
    #Threshold grayscale and create mask
    #Mask blue color channel and red color channel work out fractions.

if __name__ == '__main__':
    pathname = "W:\\GranularCharge\\pingpong\\angles\\"
    filename = "img00001.png"
    image = read_img(pathname + filename)
    find_ball(image)
    #display(image)



