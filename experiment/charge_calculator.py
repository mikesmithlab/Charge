import numpy as np

d = 0.0098 #Ball diameter  Blue acrylic and Ceramic = 8mm
d_pixels = 875-583 #Ball diameter in pixels
#d = 0.02
#d_pixels = 520
x1_pixels = 670
x2_pixels = 1088


x1 =x1_pixels *d/d_pixels#0.0996#m first point in time
x2 =x2_pixels *d/d_pixels#0.1046#m
E =2/0.13#30.81#kV/m

m=0.568 + 0.0302#g    Blue 8mm = 0.267 / Ceramic 8mm = 0.552
#m=0.035
#m=0.15
g=9.81
l=1#m


charge = m*g*(x2-x1)/(1000*l*E*1000)

print(charge, 'C')

#print((d/d_pixels)*m*g/(1000*l*E*1000))