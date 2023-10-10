from labequipment.omega_temperature_probe import Probe
from labvision.camera.quick_timer import QuickTimer
import numpy as np
from time import sleep
from datetime import datetime

time = 0
delay = 20
pathname = 'Z:/GranularCharge/BlueBeads/2023_03_20'
expt_name = 'Try1'
filename = pathname + '/' + expt_name + '_temphumidity.txt'

with Probe(port='COM3') as probe:
    for i in range(100000):
        print(time)
        humidity = probe.get_relative_humidity()
        print('Humidity =', humidity)
        temp = probe.get_temp_C()
        print('Temperature =', temp)
        sleep(delay)
        data = np.array([time, temp, humidity])
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        with open(filename, "a") as f:
            f.write(str(time) + "," + str(current_time) + "," + str(temp) + "," + str(humidity) + "\n")
        print("Current Time =", current_time)
        time = time + delay

"""
def take_measurement(t_elapsed, func_args, func_kwargs):
    print(t_elapsed)
    val = probe.get_relative_humidity()
    print('Humidity =', val)
    val = probe.get_temp_C()
    print('Temperature =', val)

stop_time = 1000000
time_gap = 10
num_measurements = int(stop_time/time_gap)

time = list(np.linspace(0, stop_time, num_measurements))

timer = QuickTimer(time_list=time, func=take_measurement)
timer.start()
"""
