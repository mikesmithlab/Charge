import numpy as np

d = 0.025 #Ball diameter
d_pixels = 955-206 #Ball diameter in pixels
x1_pixels = 583
x2_pixels = 576



x1 =x1_pixels *d/d_pixels#0.0996#m first point in time
x2 =x2_pixels *d/d_pixels#0.1046#m
E = 10/0.13#30.81#kV/m

m=7.113#g
g=9.81
l=1#m


charge = m*g*(x2-x1)/(1000*l*E*1000)

print(charge, 'C')

print((d/d_pixels)*m*g/(1000*l*E*1000))