from labvision.images import display
from labvision.video import ReadVideo
import numpy as np

from particletracker import track_gui



if __name__ == '__main__':
    pathname = 'C:/Users/ppzmis/OneDrive - The University of Nottingham/Documents/Papers/Charge/Static_Charging/Figures/Figure1/2024_02_02/'
    filename = 'PTFE_expt1_2024-02-02-145248-0000.avi'

    
    #track_gui(pathname + filename)

    vid = ReadVideo(pathname + filename)
    img = vid.read_frame(n=294)
    pts = display(img)

    print(pts)
    angle = (180/np.pi)*np.arctan((pts[1][1]-pts[0][1])/(pts[1][0] - pts[0][0]))
    print(angle)


