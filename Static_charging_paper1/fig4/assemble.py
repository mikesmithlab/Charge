"""Used to assemble images into animation of torque model"""

from labvision.video import imgs_to_video

from filehandling import smart_number_sort

path = "C:/Users/ppzmis/OneDrive - The University of Nottingham/Documents/Papers/Charge/Static_Charging/Figures/Supplementary_Vids/Torque_Model/"
imgs_to_video(path + "torque*.png", path + "model.mp4", sort=smart_number_sort)
