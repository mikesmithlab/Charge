from labvision.images import display
from labvision.video import ReadVideo
import numpy as np

from particletracker import track_gui



if __name__ == '__main__':
    pathname = 'C:/Users/ppzmis/OneDrive - The University of Nottingham/Documents/Papers/Charge/Figures/rolling_beads/2024_01_24/'
    filename = 'PP_expt1_2024-01-24-124233-0000.avi'

    
    #track_gui(pathname + filename)

    vid = ReadVideo(pathname + filename)
    img = vid.read_frame(n=1342)
    pts = display(img)

    print(pts)
    angle = (180/np.pi)*np.arctan((pts[1][1]-pts[0][1])/(pts[1][0] - pts[0][0]))
    print(angle)


