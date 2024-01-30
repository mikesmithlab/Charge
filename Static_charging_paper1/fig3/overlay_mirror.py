"""Code for generating ghost images at top of figure 3"""

from labvision.video import ReadVideo
from labvision.images import write_img, display
import cv2


def images_to_add(vid, frames, alphas):
    '''
    Returns a list of images to add to the video
    '''
    
    images = []
    
    for n in frames:
        images.append((1/255)*vid.read_frame(n).astype(float))
    
    print(len(images))

    blendedImage =  alphas[0]*images[0]
    
    for i in range(1,len(images)):
        blendedImage = blendedImage + alphas[i] * images[i]
        print(display(blendedImage))
    blendedImage = (255*blendedImage).astype('uint8')
    return blendedImage







if __name__ == '__main__':
    filename = "C:/Users/ppzmis\OneDrive - The University of Nottingham/Documents/Papers/Charge/Figures/SupplementaryVIds/Mirror_Charge_HiRes.mp4"
    vid = ReadVideo(filename)
    img = images_to_add(vid, [660, 666, 668], [0.6, 0.2, 0.2, 0.6])
    #img = images_to_add(vid, [0, 550, 662, 1000], [0.6, 0.2, 0.2, 0.6])
    display(img)
    #write_img(img, 'C:/Users/ppzmis\OneDrive - The University of Nottingham/Documents/Papers/Charge/Figures/SupplementaryVIds/Mirror_Charge_HiRes.png')

    