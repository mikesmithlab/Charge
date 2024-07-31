"""Code for generating ghost images at top of figure 3"""

from labvision.video import ReadVideo
from labvision.images import write_img, display
import cv2


def images_to_add(vid, frames, alphas):
    '''
    Returns a list of images to add to the video
    '''

    images = []

    for n in frames:
        images.append((1/255)*vid.read_frame(n).astype(float))

    print(len(images))

    blendedImage = alphas[0]*images[0]

    for i in range(1, len(images)):
        blendedImage = blendedImage + alphas[i] * images[i]
        print(display(blendedImage))
    blendedImage = (255*blendedImage).astype('uint8')
    return blendedImage


if __name__ == '__main__':
    from particletracker import track_gui
    from labvision.video import ReadVideo, WriteVideo
    filename = "C:/Users/ppzmis/OneDrive - The University of Nottingham/Documents/Papers/Charge/Static_Charging/Figures/Supplementary_Vids/SupplementaryVideo2.mp4"
    # filename = "C:/Users/ppzmis/OneDrive - The University of Nottingham/Documents/Papers/Charge/Static_Charging/Figures/test.mp4"
    # track_gui(filename)

    with ReadVideo(filename) as vid:
        print(vid.num_frames)
        img = images_to_add(vid, [1117, 2167, 2274, 2601], [
            0.75, 0.25, 0.25, 0.75])
    # img = images_to_add(vid, [660, 666, 668], [0.6, 0.2, 0.2, 0.6])

    display(img)
    # vid.close()
    write_img(img, filename[:-4] + '.png')
