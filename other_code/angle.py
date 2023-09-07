from labvision.images import display
from labvision.video import ReadVideo
from labvision.images import display
import numpy as np

from filehandling import open_filename

filename = open_filename('E:/Mike')

#pathname = 'E:/Mike/2023_04_17/'
#filename = 'P1001593.mp4' #frame = 3559
#filename = 'P1001595.mp4' #frame = 2628
#filename = 'P1001601.mp4' #frame=


vid = ReadVideo(filename)

frame_num = 1000

while True:
    display(vid.read_frame(n=frame_num))
    val = input('new frame:')
    if val == 'q':
        break
    else:
        frame_num = int(val)

img = vid.read_frame(n=frame_num)
#Click pivot and bead and then top and bottom of glass slide
pts = display(img)
tan_theta = (pts[0][0]-pts[1][0])/(pts[0][1]-pts[1][1])
#tan_slide_angle = (pts[2][0]-pts[3][0])/(pts[2][1]-pts[3][1])
#tan_theta = np.tan(np.arctan(tan_theta)-np.arctan(tan_slide_angle))

m = (0.267+0.0302)*1E-3
g = 9.807
F = m*g*tan_theta
print(pts)
print('Force: {}'.format(F))