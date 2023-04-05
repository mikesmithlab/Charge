from filehandling import BatchProcess
import pandas as pd
import matplotlib.pyplot as plt



filter = 'Z:\\GranularCharge\\siliconeballs\\2023_03_06_stiffness\\bead*.csv'
filter2 = 'Z:\\GranularCharge\\siliconeballs\\2023_03_06_stiffness\\baked*.csv'
plt.figure(1)
bead_numbers = []
for file in BatchProcess(filter):
    print(file)
    bead_number = file.split('\\')[-1][4:6].replace('_','').replace('r','').replace('.','')
    bead_numbers.append(bead_number)
    df=pd.read_csv(file)
    gap = df['Gap(mm)'].to_numpy()
    force = df['Normal force(N)'].to_numpy()
    plt.plot(gap,force, 'b.')

for file in BatchProcess(filter2):
    print(file)
    bead_number = file.split('\\')[-1][4:6].replace('_','').replace('r','').replace('.','')
    bead_numbers.append(bead_number)
    df=pd.read_csv(file)
    gap = df['Gap(mm)'].to_numpy()
    force = df['Normal force(N)'].to_numpy()
    plt.plot(gap,force, 'r.')   

plt.show()
    
