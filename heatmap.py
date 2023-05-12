from labvision.images import display, gaussian_blur, draw_circle, WHITE, mask, threshold, find_connected_components, extract_biggest_object
from labvision.video import ReadVideo
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import cv2
from matplotlib.colors import LogNorm
from pylab import figure, cm


import numpy as np


#P1001621 => crop_xy = ((471, 20), (2612, 2118)), mask_xy = ((1060, 1043), (1869, 1453))

def crop_mask_img(img, crop_xy = ((551, 60), (2530, 2070)), mask_xy = ((992, 992), (1671, 1578))):
    """Crop and mask img"""
    cropped_img = threshold(gaussian_blur(img[crop_xy[0][1]:crop_xy[1][1],crop_xy[0][0]:crop_xy[1][0],:]), value=86, mode=cv2.THRESH_BINARY_INV)
    
    cx = (crop_xy[1][0]-crop_xy[0][0])/2
    cy = (crop_xy[1][1]-crop_xy[0][1])/2
    r = (mask_xy[1][0]-mask_xy[0][0])/2

    mask_img = np.zeros(np.shape(cropped_img),cropped_img.dtype)
    mask_img = draw_circle(mask_img, cx, cy, r, color=WHITE, thickness=-1)[:,:,0]
    
    img = mask(cropped_img[:,:,2], mask_img)
    return img


def find_centroid(new_img):
    """Find centre of bead"""
    _,stats,centroids=find_connected_components(new_img)

    areas = stats[:,4]
    try:
        particle_index = np.argwhere((areas >10000) & (areas < 100000))[0][0]
    except:
        display(new_img)

    cx=int(centroids[particle_index][0])
    cy=int(centroids[particle_index][1])
    return cx, cy
    
def analyse_vid(vid):
    
    for i, img in enumerate(vid):
        new_img=crop_mask_img(img)
        cx,cy=find_centroid(new_img)
        if i==0:
            data = [[cx,cy]]
        else:
            data.append([cx,cy])
    data=np.array(data)
    np.savetxt(path+filename[:-4] + '_heatmap.txt', data)
    return data


def generate_heatmap(data, start=0, stop=1, shape=(1000,1000), rad=5):
    heatmap = np.zeros(shape)
    
    mask = np.zeros((2*rad+1,2*rad+1))
    centre = rad
    x=np.arange(0,2*rad+1,1)
    y=np.arange(0,2*rad+1,1)

    for i in x:
        for j in y:
            dist = np.sqrt((i-centre)**2 + (j-centre)**2)
            if dist <=rad:
                mask[i,j] +=1
    
    data = data[start:stop+1]
    x=np.arange(0, shape[0])
    y=np.arange(0, shape[1])

    for point in data:
        cx = int(point[0])
        cy= int(point[1])
        
        heatmap[cx-rad:cx+rad+1,cy-rad:cy+rad+1] = heatmap[cx-rad:cx+rad+1,cy-rad:cy+rad+1] +  mask
    
    return heatmap

def plot_heatmap(heatmap, rad_px : float):
    print('num_pixels:')
    num_pixels = np.count_nonzero(heatmap)
    print(num_pixels)

    f = plt.figure(figsize=(6.2, 5.6))
    ax = f.add_axes([0.1, 0.1, 0.72, 0.72])
    axcolor = f.add_axes([0.90, 0.1, 0.02, 0.72])

    im = ax.matshow(heatmap, cmap='jet', norm=LogNorm(vmin=0.01, vmax=np.max(np.max(heatmap))))

    cx,cy = np.unravel_index(np.argmax(heatmap),np.shape(heatmap))
    circ = patch.Circle((int(cx),int(cy)), radius=rad_px, fill=False, color='k', linestyle='--')
    ax.add_patch(circ)

    #t = [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    f.colorbar(im, cax=axcolor, format="%.2f")
    

if __name__ == '__main__':
    #from particletracker import track_gui
    path = 'E:\\RawData\\Mike\\2023_05_02\\'
    filename = 'P1001621.mp4'

    #track_gui(path + filename)

    start = 0
    stop = None
    bead_diam = 7.8E-3


    vid = ReadVideo(path+filename, frame_range=(start, stop, 1))
    img=crop_mask_img(vid.read_frame())
    
    scale=np.shape(img)[0]/100E-3

    analyse_vid(vid)

    data = np.loadtxt(path+filename[:-4]+'_heatmap.txt')
    print(data)
    if stop is None:
        stop = vid.num_frames
    heatmap = generate_heatmap(data, start, stop, shape=np.shape(img))
    plot_heatmap(heatmap, bead_diam * scale)
    plt.show()





