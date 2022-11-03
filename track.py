from particletracker import track_gui, batchprocess
from assemble_2_mp4 import make_movie
from plot_charge_time import plot_charge

expt_name = '2021_11_23_positive_negative_2'

path ='Z:/GranularCharge/ChargeProject/'
moviefilename = expt_name + '/' + expt_name + '.mp4'
settings_file = 'processing.param'

#This opens the gui
#track_gui(movie_filename=path + moviefilename, settings_filename=path + settings_file)

#This runs the full process and does the tracking headless
make_movie(expt_name, pathstub=path)
print('tracking...')
#batchprocess(path + moviefilename, path + settings_file)
#plot_charge(expt_name, path=path)

