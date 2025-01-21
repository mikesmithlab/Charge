import numpy as np
import matplotlib.pyplot as plt
import cv2
import labvision.images as im
from labvision.video import ReadVideo
from filehandling import BatchProcess
from scipy.signal import savgol_filter

def calc_angle(img, band_width=5, factor=0.3, threshold=30, orange_right=True, show_checks=False):
    bwimg = im.threshold(img[:,:,1], threshold)
    if show_checks:
        im.display(bwimg)
    objs,stats, centroids = im.find_connected_components(bwimg)
    sort_vals = np.argsort(stats[:,4])[::-1]
    centroids = centroids[sort_vals]
    y_mid = int(centroids[1][1])


    
    whole_ball =  smooth_data(np.mean(img[y_mid-band_width:y_mid+band_width,:,2],axis=0))
    orange_ball = smooth_data(np.mean(img[y_mid-band_width:y_mid+band_width,:,0],axis=0))

    if show_checks:
        img[y_mid - band_width:y_mid + band_width, :, :] = (255, 0, 0)
        im.display(img)

    base_line_whole = np.mean(whole_ball[:200])
    base_line_orange = np.mean(orange_ball[:200])
    shift_whole_ball =whole_ball - base_line_whole
    shift_orange_ball = orange_ball - base_line_orange
    norm_whole_ball = shift_whole_ball / np.max(shift_whole_ball)
    norm_orange_ball = shift_orange_ball / np.max(shift_orange_ball)
    diff = (norm_whole_ball - norm_orange_ball)




    if orange_right:
        ball_edge_L = np.argwhere(norm_orange_ball > factor)[0]
        ball_edge_R = np.argwhere(norm_whole_ball > factor)[-1]
        orangeline = np.argwhere(diff > factor)
        line = orangeline[0]
    else:
        ball_edge_L = np.argwhere(norm_whole_ball > factor)[0]
        ball_edge_R = np.argwhere(norm_orange_ball > factor)[-1]
        orangeline = np.argwhere(diff < factor)
        line = orangeline[0]
    pixel_no = np.linspace(0,np.size(whole_ball)-1,np.size(whole_ball))

    if show_checks:

        plt.figure()
        plt.plot(pixel_no,norm_whole_ball,'r-')
        plt.plot(pixel_no,norm_orange_ball,'b-')
        plt.plot(pixel_no,diff,'y-')
        plt.plot([ball_edge_L,ball_edge_L],[0,1],'g--')
        plt.plot([line,line],[0,1],'g--')
        plt.plot([ball_edge_R,ball_edge_R],[0,1],'g--')
        plt.plot(pixel_no[orangeline],norm_orange_ball[orangeline],'g-')
        plt.show()

    x_over_2r = (line-ball_edge_L)/(ball_edge_R - ball_edge_L)
    theta = np.arccos(x_over_2r*2 - 1)*180/np.pi
    #if theta < 85:
    #    calc_angle(img, show_checks=True)

    return theta
    
def smooth_data(intensity, window_size=17, polynomial=2):
    smooth_intensity = savgol_filter(intensity, window_size, polynomial)
    return smooth_intensity

if __name__ == '__main__':
    
    
    path = 'Z:\\GranularCharge\\pingpong\\angles\\2022_08_23\\'
    orange_right = True
    
    theta_vals = []

    #For folder of images
    """file_filter = 'img*.png'    
    for file in BatchProcess(path + file_filter):
        img = im.read_img(file)
        theta = calc_angle(img, show_checks=False)
        theta_vals.append(theta[0])
    """

    #For movie
    video_name ='purgeonoff.avi'
    print(path+video_name)
    fps=0.75
    readvid = ReadVideo(path + video_name)
    for i, img in enumerate(readvid):
        if i%1000 == 0:
            print(readvid.num_frames - i)

        try:
            theta = calc_angle(img, show_checks=False, orange_right=orange_right)
            theta_vals.append(theta[0])
        except IndexError:
            print('error')
            pass#theta = calc_angle(img, show_checks=True, orange_left=orange_right)


    theta_vals = np.array(theta_vals)
    t = (1/fps)*np.linspace(0,np.size(theta_vals)-1,np.size(theta_vals))
    np.savetxt(path + video_name[:-4] + '.txt', theta_vals)

    plt.figure()
    plt.plot(t,theta_vals,'r-')
    plt.plot(t,theta_vals,'b.')
    plt.xlabel('t (s)')
    plt.ylabel('angle (deg)')
    plt.savefig(path + video_name[:-4] + '.png')
    plt.show()
    