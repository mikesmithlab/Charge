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
    
def find_bead_and_tag(img, mask_top=200, th1=26, th2=162, configure=False,**kwargs):
    """Extract bindary images of just bead and just tag for later processing"""
    bead = threshold(cv2.subtract(img[:,:,0],img[:,:,2]), value=th1, configure=configure)
    tag = threshold(img[:,:,2], value=th2, configure=configure) 
    tag[:mask_top,:]=0 
    return bead, tag

def find_bead_xy(bead, width_bead):
    """Find centre of bead"""
    cntrs = find_contours(bead)
    area = []
    for cnt in cntrs:
        area.append(cv2.contourArea(cnt))
    area=np.array(area)
    index = np.argmin((area-3.14*(width_bead/2)**2)**2)
    info = rotated_bounding_rectangle(cntrs[index])
    return info

def find_tag(tag, bead_info, bead_px):
    #Uses vertical box
    contours = find_contours(tag, hierarchy=False)
    beadx = bead_info['cx']
    beady = bead_info['cy']

    reduced_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if (area < (2*bead_px**2)) & (area > 0.05*bead_px**2):
            M = cv2.moments(cnt)
            cX =int(M["m10"]/M["m00"])
            cY = int(M["m01"]/M["m00"])
            if (cX - beadx)**2 < (0.5*bead_px)**2:
                #cnt=cv2.convexHull(cnt)
                reduced_contours.append(cnt)
                
    tag_contour = sort_contours(reduced_contours)[-1]
    M = cv2.moments(tag_contour)
    cX =M["m10"]/M["m00"]
    cY = M["m01"]/M["m00"]

    xvals = []

    min_value_x = + int(cX) - 2*bead_px

    width_slice = np.sum(tag[int(cY)-2:int(cY) + 2, min_value_x:int(cY) + 2*bead_px],axis=0)

    
    indices = np.where(width_slice > 100) 
    
    minx = np.min(indices) + min_value_x
    maxx = np.max(indices) + min_value_x
    width = minx-maxx

    angle = np.arctan2((cX-beadx),(cY-beady))
    #print('angle : {}'.format(angle))
    #rot_contour = rotate_contour(tag_contour, angle)

    (x,y,w,h) = cv2.boundingRect(tag_contour)      

    rec = (
        (cX, cY), # center pt
        (w, h),  # W, H
        0                        # angle
        )
    box = cv2.boxPoints(rec)

    tag_info = {'width':w,'angle':angle,'cx': cX,'cy': cY, 'box':box, 'contour':tag_contour, 'minx':minx,'maxx':maxx, 'width':width}

    return tag_info

def annotate_img(img, tag_info, bead_info, diam_px):

    annotated_img = draw_polygon(img.copy(),tag_info['box'],thickness=2)
    annotated_img = draw_circle(annotated_img, int(bead_info['cx']), int(bead_info['cy']), int(diam_px/2))
    annotated_img = cv2.line(annotated_img, (int(bead_info['cx']),int(bead_info['cy'])),(int(tag_info['cx']),int(tag_info['cy'])),(255,255,0),3)
    annotated_img = cv2.line(annotated_img, (int(tag_info['minx']),int(tag_info['cy'])),(int(tag_info['maxx']),int(tag_info['cy'])),(255,0,255),3)
    annotated_img = draw_contours(annotated_img, [tag_info['contour']] )
    pts=display(annotated_img)

def get_data(tag_info, bead_info, scale):
    return [bead_info['cx'],bead_info['cy'], tag_info['width'], tag_info['angle'], get_theta(tag_info['width'], scale=scale)]

def get_theta(Width_px, t=0.7E-3, W=7.7E-3, scale=1):
    theta = (np.pi/180)*np.linspace(0,89.9, 1800)
    L = np.abs(W*np.cos(theta))+np.abs(t*np.sin(theta))
    index = np.argmin(np.abs(scale*Width_px - L))
    theta_measured = theta[index]*180/np.pi
    return theta_measured



if __name__=='__main__':

    

    """Notes - 4 scenarios:
        1.  Measuring angle of tag in field.
        2.  Measuring oscillation
        3.  Measuring approach to plate
        4. Measuring unwinding
        
        1 & 2 use the ximea camera 3 & 4 the Panasonic       
        1. Is an image sequence the rest are movies
        

        """
    
    #Approaching plate
    pathname = 'E:\\RawData\\Mike\\2023_04_19\\'
    filename = 'P1001614.mp4'
    
    #In the field
    #pathname = 'Z:\\GranularCharge\\BlueBeads7mm\\2023_05_04\\'
    #filename = 'img*.png'

    #from particletracker import track_gui
    #track_gui(pathname + filename)

    #Oscillation
    #pathname = 'C:\\Users\\ppzmis\\OneDrive - The University of Nottingham\\Documents\\Papers\\Charge\\ChargeProjectProcessedData\\torsional_spring_constant\\'
    #filename = '2023_04_06_bluebeadoscillation2.avi'

    readVid = ReadVideo(pathname + filename, frame_range=(28,None,1))
    
    img=readVid.read_frame(n=28)
    readVid.set_frame(n=4000)
    scale, diam_px, height_px = get_scale(img)
    

    #Approaching plate
    params = {'mask_top':200, 'th1':26, 'th2':162, 'scale':scale, 'width_bead': diam_px, 'configure':True}

    #In the field
    #params = {'mask_top':0, 'th1':18, 'th2':84, 'scale':scale, 'width_bead': diam_px, 'configure': False}
    


    
    df = pd.DataFrame(columns=['beadx','beady','tag_proj_width','tag_angle_vertical', 'tag_rotation_angle'], index=range(0,readVid.num_frames, 1))

    for i,img in enumerate(readVid):  
                    
        bead, tag = find_bead_and_tag(img, **params)#, min_rad=int(diam_px/2-5) ,max_rad=int(diam_px/2+5))
        bead_info=find_bead_xy(bead, diam_px)
        tag_info=find_tag(tag, bead_info, diam_px)
        if i%1000==0:
            print(i)
            annotate_img(img.copy(), tag_info, bead_info, diam_px)
        df.loc[i] = get_data(tag_info, bead_info, scale)

    try:
        pass#df.to_csv(pathname + filename[:-4] + '.csv')
    except:
        pass#df.to_csv(pathname + filename[:-5] + '.csv')
    """
    framerate = 30

    t=np.array(list(range(len(width))))/framerate
    width=np.array(width)
    theta=np.array(theta)

    np.savetxt(pathname+filename[:-4]+'.csv',np.c_[t,width,theta], delimiter=',',encoding='utf-8')

    plt.figure(1)
    plt.plot(t,width,'r.')
    plt.plot(t,width,'b-')
    plt.title('projected width')
    plt.xlabel('Times (s)')
    plt.ylabel('Width (m)')

    plt.figure(2)
    plt.plot(t,theta,'b-')
    plt.plot(t,theta,'g.')
    plt.title('angular fluctuation')
    plt.xlabel('Times (s)')
    plt.ylabel('Angle (deg)')
    plt.show()

    """






