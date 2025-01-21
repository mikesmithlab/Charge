import numpy as np
from labequipment import daq
from time import sleep


def strike_ntimes(amplitude=0.1, freq=1,num_times=1):
    aio = daq.analog(reset=True)
    aio.addOutput(1)
    one=np.ones(1000)
    aio.Rate = freq*2000
    signal = np.append(amplitude * one, -amplitude * one)

    timeout = 0.2 + np.size(signal) / 1000
    aio.write(signal, continuous=True)
    aio.tAO.StopTask()
    sleep(timeout)

    for i in range(num_times):
        aio.tAO.StartTask()
        sleep(timeout)
        aio.tAO.StopTask()

if __name__ == '__main__':
    strike_ntimes(amplitude=0.3, num_times=10)








print('get ready...')




#aio.stop()
