import numpy as np
import matplotlib.pyplot as plt
import cv2
import labvision.images as im
from labvision.video import ReadVideo
from filehandling import BatchProcess


def calc_angle(img, band_width=5, factor=1.2, show_checks=False):
    bwimg = im.threshold(img[:,:,1], 120)
    
    objs,stats, centroids = im.find_connected_components(bwimg)
    sort_vals = np.argsort(stats[:,4])[::-1]
    centroids = centroids[sort_vals]
    y_mid = int(centroids[1][1])
    
    
    whole_ball =  np.mean(img[y_mid-band_width:y_mid+band_width,:,2],axis=0)
    orange_ball = np.mean(img[y_mid-band_width:y_mid+band_width,:,0],axis=0)
    

    base_line_whole = np.mean(whole_ball[:200])
    base_line_orange = np.mean(orange_ball[:200])
    ball_edge = np.argwhere(whole_ball > base_line_whole *factor)
    ball_edge_L = ball_edge[0]
    ball_edge_R = ball_edge[-1]
    line = np.argwhere(orange_ball > base_line_orange * factor)[0]
    
    if show_checks:
        im.display(bwimg)
        plt.figure()
        plt.plot(whole_ball,'r-')
        plt.plot(orange_ball,'b-')
        plt.plot([ball_edge_L,ball_edge_L],[0,1000],'g--')
        plt.plot([line,line],[0,1000],'g--')
        plt.plot([ball_edge_R,ball_edge_R],[0,1000],'g--')
            
        img[y_mid-band_width:y_mid+band_width,:,:] = (255,0,0)
        im.display(img)
        plt.show()

    x_over_2r = (line-ball_edge_L)/(ball_edge_R - ball_edge_L)
    theta = np.arccos(x_over_2r*2 - 1)*180/np.pi
    return theta
    


if __name__ == '__main__':
    
    
    path = 'W:\\GranularCharge\\pingpong\\angles\\2022_07_20\\'
    
    
    theta_vals = []

    #For folder of images
    file_filter = 'img*.png'    
    for file in BatchProcess(path + file_filter):
        img = im.read_img(file)
        theta = calc_angle(img, show_checks=False)
        theta_vals.append(theta[0])

    #For movie
    """video_name = 'example.mp4'
    for img in ReadVideo(path + video_name):
        theta = calc_angle(img, show_checks=False)
        theta_vals.append(theta[0])
    """

    theta_vals = np.array(theta_vals)
    np.savetxt(path + 'theta.txt', theta_vals)
    