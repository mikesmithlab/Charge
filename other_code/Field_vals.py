import pandas as pd
import matplotlib.pyplot as plt

expt_name = '21_10_05_added2ndbead'
plate_separation = 0.13#m
bead_diameter = 0.01#m





path = 'Z:/GranularCharge/ChargeProject/' + expt_name + '/'
file = expt_name + '.hdf5'

df_total= pd.read_hdf(path + file)
df = df_total[df_total['classifier'] == True]

bead_diameter_pixels = 2*df['r'].mean()
scale = bead_diameter/bead_diameter_pixels

df['E'] = df['voltage'] / plate_separation

xpos = df['x'].to_numpy()*scale
E = df['E'].to_numpy()
frames = df['frame'].to_numpy()

plt.figure(1)
plt.plot(frames, xpos,'rx')
plt.ylabel('x position (m)')
figure, axis_1 = plt. subplots()
axis_1.set_xlabel('frame number')
axis_1.set_ylabel('Xpos')
axis_2 = axis_1.twinx() #create second axis
axis_1.plot(frames, xpos,'rx')
axis_2.plot(frames, E, 'bx')
plt.show()

