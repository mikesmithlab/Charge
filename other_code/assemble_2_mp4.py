from labvision.video.opencv_io import WriteVideo
from labvision.images import read_img
from filehandling import BatchProcess, smart_number_sort
import numpy as np


def sort_ximea_camfiles(filenames):
    filename_sort = []
    # Secondary sort criterion is the numerical value
    for filename in filenames[:-1]:

        filename_sort.append(filename.split('(')[1].split(')')[0])
        #filename_sort.append(''.join([i for i in filename if i in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']]))

    # Sort by length of number first. This means 01 goes before 001.
    len_filenames = [len(number) for number in filename_sort]

    sorted_filenames = [x for _, _, x in sorted(zip(len_filenames, filename_sort, filenames))]
    sorted_filenames.insert(0, filenames[-1])
    print(sorted_filenames)
    return sorted_filenames





def make_movie(expt_name, pathstub='Z:/GranularCharge/ChargeProject/'):
    print('Assembling images into movie')
    print(expt_name)
    path =  pathstub + expt_name + '/'
    filefilter = 'img_*.png'

    output_vid_name = path + expt_name + '.mp4'
    print(output_vid_name)
    writevid = WriteVideo(output_vid_name, frame_size=(1024, 1280, 3))

    for filename in BatchProcess(path + filefilter, smart_sort=smart_number_sort):
         print(filename)
         img = read_img(filename)
         writevid.add_frame(img)


    writevid.close()

if __name__ == '__main__':
    expt_name = '2021_11_11_PTFE'
    make_movie(expt_name)