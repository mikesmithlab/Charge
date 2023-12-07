import pandas as pd
import numpy as np
import os
import io
import decimal

def convert_frame(data, ball_rad=5E-3, patch_rad_ratio=0.95, surface_rad=1E-3, max_charge=0):
    data.columns=['frame', 'particle_type', 'xc', 'yc', 'zc', 'charge']

    xb = data['xc'].loc[0]
    yb = data['yc'].loc[0]
    zb = data['zc'].loc[0]

    #This will hold the adjusted coords to account for patch sphere radius so that that patch sphere touches original surface of ball
    data['xmod'] = data['xc']
    data['ymod'] = data['yc']
    data['zmod'] = data['zc']

    #Set rads of data
    
    data['rad'] = surface_rad
    data.loc[data['particle_type']==0,'rad'] = ball_rad
    data.loc[data['particle_type']==1,'rad'] = ball_rad*patch_rad_ratio
    
    #Sets transparency of data
    data['transparency']=0.0
    data.loc[data['particle_type']==0,'transparency'] = 0.5

    #scale_charge
    print(0.5*(1+(data.loc[data['particle_type']==1,'charge'])/max_charge))
    data['update_charge'] = 0.5 #0.5 is neutral
    data.loc[data['particle_type']==1,'update_charge'] = 0.5*(1-(data.loc[data['particle_type']==1,'charge'])/max_charge) # ball charges are negative
    data.loc[data['particle_type']==2,'update_charge'] = 0.5*(1+(data.loc[data['particle_type']==2,'charge'])/max_charge) # surface charges are positive


    # Figure out the vector and shift patch sphere centres
    data.loc[data['particle_type']==1,'xmod'] = data.loc[data['particle_type']==1,'xc'] - patch_rad_ratio*(data.loc[data['particle_type']==1,'xc'] - xb)
    data.loc[data['particle_type']==1,'ymod'] = data.loc[data['particle_type']==1,'yc'] - patch_rad_ratio*(data.loc[data['particle_type']==1,'yc'] - yb)
    data.loc[data['particle_type']==1,'zmod'] = data.loc[data['particle_type']==1,'zc'] - patch_rad_ratio*(data.loc[data['particle_type']==1,'zc'] - zb)

    # Shift surface particles down by radius of surface charges.
    data.loc[data['particle_type']==2,'zmod'] = data.loc[data['particle_type']==2,'zc'] - surface_rad
    return data


def ovito_prep(input_filename, output_filename, num_ball_charges=27, num_surface_charges=400):
    
    num_particles = num_surface_charges+num_ball_charges + 1
    max_charge=0

    with open(input_filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if (line[0] != 't') and (line[:3] != str(num_particles)):
                if float(line.split(',')[5]) > max_charge:
                    max_charge = float(line.split(',')[5])
    
    print('max_charge = ', max_charge)
    
    num_frames = len(lines)/(num_particles+2)

    
    if os.path.exists(output_filename):
        os.remove(output_filename)

    for i in range(int(num_frames)):
        df=pd.read_csv(input_filename, skiprows=1+i*(num_particles+2), nrows=num_particles, float_precision='high')
        df = convert_frame(df, max_charge=max_charge)
        
        with open(output_filename, 'a') as f:
            if i!=0:
                f.write('\n')
            f.write(str(num_particles) + '\n')
            f.write(df.to_string(index=False))


if __name__ == '__main__':
    ovito_prep('test.dat', 'test_ovito.dat')