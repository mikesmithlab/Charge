from labvision.images import read_img, write_img, display
import numpy as np

path = 'Z:/GranularCharge/ChargeProject/2021_10_20_beadoncotton_decay/'

filename_img1 = path + 'img_001.png'
filename_img2 = path + 'img_002.png'

print(filename_img1)
img1 = read_img(filename_img1)
img2 = read_img(filename_img2)

#display(img1)
print(np.shape(img1))
new_img = np.zeros(np.shape(img1),dtype=np.uint8)

split_val = 650
split_val2 = 1000


new_img[:,split_val:,:] = img2[:,split_val:,:]
new_img[:,:split_val,:] = img1[:,:split_val,:]
#new_img[:,split_val2:,:] = img2[:,split_val2:,:]

display(new_img)

write_img(new_img, path+'imgbkg.png')