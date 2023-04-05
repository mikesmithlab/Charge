import numpy as np
from filehandling import BatchProcess

#bkg should have 1000 points
#raw should have 15000 points
def convert_file(filename):
    with open(filename, 'r') as fid:
        try:
            data=fid.readline()

            no_skip=True
            if 'bkg' in filename:
                numpts = 1000
            elif 'raw' in filename:
                numpts = 15000
            else:
                no_skip = False

            if no_skip:
                i=0
                t=0
                new_data=[]
                time_data=[]
                time=0
                dt=0.001
                while i < len(data):
                    if data[i] == '-':
                        new_data.append(data[i:i+9])
                        i=i+9
                    else:
                        new_data.append(data[i:i+8])
                        i=i+8
                    time_data.append(time)
                    time=time+dt

       
                current_vals = np.array(new_data[:numpts],dtype='float')
                time_array = np.array(time_data[:numpts])

                np.savetxt(filename[:-4]+'_corrected.txt',np.c_[time_array, current_vals])
        except:
            pass

path = "C:\\Users\\ppzmis\\OneDrive - The University of Nottingham\\Documents\\Work\\Programming\\Granular\\"
filefilters = [ path + "2021_01_26_redpprepeat_charging\\*.txt",
                path + "2021_01_26_redpprepeat_uncharged_reference\\*.txt",
               ]


for filefilter in filefilters:
    for filename in BatchProcess(filefilter):
        convert_file(filename)