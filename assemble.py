from labvision.video import imgs_to_video

from filehandling import smart_number_sort

path="C:/Users/mikei/OneDrive - The University of Nottingham/Documents/Papers/Charge/Figures/FIgure2/torque_model/"
imgs_to_video(path + "torque*", path + "model.mp4", sort=smart_number_sort)


