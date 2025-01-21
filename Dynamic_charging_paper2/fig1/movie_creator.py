from labvision.video import ReadVideo, WriteVideo
from labvision.images.cropmask import viewer, crop
from labvision.images.basics import display
import numpy as np
from matplotlib.pyplot import imshow, show


def combine(img1, img2, rect1 = ((650, 3), (1200, 703)), rect2 = ((363, 109), (863, 809)), scale=3):
    return np.hstack((crop(img2, rect2)[::scale,::scale,:], crop(img, rect1)[::scale,::scale,:]))



if __name__ == '__main__':
    path =r"C:/Users/ppzmis/OneDrive - The University of Nottingham/Documents/Papers/Charge/Dynamic_Charging/Figures/Supplementary_movie1/"
    vid1 = "P1002203.mp4"
    vid2 = "coloured_dots_2024-11-19-121122-0000.avi"

    start_frame_1 = 0 # side vid
    start_frame_2 = 150 # Top vid
    

    readvid1 = ReadVideo(path + vid1)
    readvid2 = ReadVideo(path + vid2)
    delta_end_frame = min(readvid1.num_frames, readvid2.num_frames)- max(start_frame_1, start_frame_2) - 10


    img = readvid1.read_frame(n=start_frame_1)
    img2 = readvid2.read_frame(n=start_frame_2)
    print(display(img))
    print(display(img2))
    test_img = combine(img,img2)
    display(test_img)
    
    writevid = WriteVideo(path + 'output.mp4', frame=test_img, fps=30.0)
    
    for idx in range(delta_end_frame):    
        img = readvid1.read_frame(n=start_frame_1+idx)
        img2 = readvid2.read_frame(n=start_frame_2+idx)
        new_img = combine(img, img2)
        writevid.add_frame(new_img)   
    
    
    readvid1.close()
    readvid2.close()
    writevid.close()

