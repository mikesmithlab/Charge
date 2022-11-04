import numpy as np

d = 0.012 #Ball diameter
d_pixels = 1091-326 #Ball diameter in pixels
x1_pixels = 713
x2_pixels = 541



x1 =x1_pixels *d/d_pixels#0.0996#m first point in time
x2 =x2_pixels *d/d_pixels#0.1046#m
E = 10/0.13#30.81#kV/m

m=0.977#g
g=9.81
l=1#m


charge = m*g*(x2-x1)/(1000*l*E*1000)

print(charge, 'C')

print((d/d_pixels)*m*g/(1000*l*E*1000))