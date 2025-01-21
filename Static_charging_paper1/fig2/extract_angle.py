
"""Notes


Analysis of the dipole comes in two parts:

1. Extract angles from images in a folder (this script). These then need manually splitting up into experiments
2. Fit the curve to the dipole model (analyse_charge_dipole.py). This requires you to create a file of electric field values
"""

from labvision.video import ReadVideo
from labvision.images.basics import display
from labvision.images.thresholds import threshold
from labvision.images import find_contours, sort_contours, draw_polygon, draw_circle, find_contours, sort_contours, rotated_bounding_rectangle, get_shape

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_scale(img, width=9.8E-3):
    """Work out image scale from the diameter of the ball"""
    coords = display(img)
    width_pixels = np.abs(coords[1][0] - coords[0][0])
    scale = width / width_pixels
    height_px = int(coords[0][1])
    print('Scale is : {}'.format(scale))
    print('Width_pixels : {}'.format(width_pixels))
    return scale, width_pixels, height_px


def extract_bead_and_tag(img, mask_top=10, th1=26, configure=False, **kwargs):
    """Extract bindary images of just bead and just tag for later processing"""
    bead_and_tag = threshold(img[:, :, 2], value=th1, configure=configure)

    # mask top of image so contour doesn't connect with edge of img.
    bead_and_tag[:mask_top, :] = 0
    return bead_and_tag.astype(np.uint8)


def get_largest_contour(img, width_bead):
    """Get largest contour from image
    img : binary image
    width_bead : width of bead in pixels
    """
    contours = find_contours(img, hierarchy=False)
    contours = sort_contours(contours)[::-1]

    # May need to delete the frame as the largest contour
    if cv2.contourArea(contours[0]) > (4 * width_bead**2):
        contours = contours[1:]
    contour = contours[0]
    return contour


def com_contour(contour):
    M = cv2.moments(contour)
    cX = M["m10"] / M["m00"]
    cY = M["m01"] / M["m00"]
    return cX, cY


def find_bead_xy(img, width_bead):
    """Find centre of bead"""
    contour = get_largest_contour(img, width_bead)
    info = rotated_bounding_rectangle(contour)

    max_x = 0
    min_x = 10000
    centre_y = 0

    comX, comY = com_contour(contour)

    for pt in contour:
        pt = pt[0]
        if (pt[0] > max_x) and (pt[1] > comY):
            max_x = pt[0]
            centre_y = pt[1]
        if (pt[0] < min_x) and (pt[1] > comY):
            min_x = pt[0]

    info['cx'] = (max_x + min_x) / 2
    info['cy'] = centre_y

    return info


def mask_bead(img, bead_info, width_bead):
    """masks the section of the image with the bead in so we can find the tag

    Parameters
    ----------
    img : _type_
        binary img
    bead_info : _type_
        dictionary like {'cx': , 'cy':}
    """
    img[bead_info['cy'] - int(width_bead / 2):, :] = 0
    return img


def find_tag(img, bead_info, width_bead):
    """Find the tag in the image"""
    img = mask_bead(img, bead_info, width_bead)
    contours = find_contours(img, hierarchy=False)

    beadx = bead_info['cx']
    beady = bead_info['cy']

    reduced_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if (area < (2 * width_bead**2)) & (area > 0.05 * width_bead**2):
            cX, cY = com_contour(cnt)
            if (cX - beadx)**2 < (0.5 * width_bead)**2:
                # cnt=cv2.convexHull(cnt)
                reduced_contours.append(cnt)

    tag_contour = sort_contours(reduced_contours)[-1]
    M = cv2.moments(tag_contour)
    cX = M["m10"] / M["m00"]
    cY = M["m01"] / M["m00"]

    min_value_x = int(cX) - width_bead
    max_value_x = int(cX) + width_bead

    if min_value_x < 0:
        min_value_x = 0
    if max_value_x > get_shape(bead_and_tag)[1]:
        max_value_x = get_shape(bead_and_tag)[1] - 1

    width_slice = np.sum(
        img[int(cY) - 10:int(cY) + 10, min_value_x:max_value_x], axis=0)

    indices = np.where(width_slice > 100)

    minx = np.min(indices) + min_value_x
    maxx = np.max(indices) + min_value_x
    width = maxx - minx

    angle = np.arctan2((cX - beadx), (cY - beady))

    (_, _, w, h) = cv2.boundingRect(tag_contour)

    rec = (
        (cX, cY),  # center pt
        (w, h),  # W, H
        0                        # angle
    )
    box = cv2.boxPoints(rec)

    tag_info = {'width': np.abs(width), 'angle': angle, 'cx': cX, 'cy': cY,
                'box': box, 'contour': tag_contour, 'minx': minx, 'maxx': maxx}

    return tag_info


def annotate_img(img, tag_info, bead_info, diam_px):
    """Draws on images all the features that are measured."""

    annotated_img = draw_polygon(img.copy(), tag_info['box'], thickness=2)
    annotated_img = draw_circle(annotated_img, int(
        bead_info['cx']), int(bead_info['cy']), int(diam_px / 2), color=(0, 255, 0), thickness=2)
    # annotated_img = cv2.line(annotated_img, (int(bead_info['cx']), int(
    #    bead_info['cy'])), (int(tag_info['cx']), int(tag_info['cy'])), (255, 255, 0), 3)
    annotated_img = cv2.line(annotated_img, (int(tag_info['minx']), int(
        tag_info['cy'])), (int(tag_info['maxx']), int(tag_info['cy'])), (255, 0, 255), 3)
    # annotated_img = draw_contours(
    #    annotated_img, [tag_info['contour']], thickness=2)
    pts = display(annotated_img)


def get_data(tag_info, bead_info, scale):
    theta_low, theta_high = get_theta(
        tag_info['width'], tag_info['angle'], scale=scale)
    return [bead_info['cx'], bead_info['cy'], bead_info['cx'] * scale, bead_info['cy'] * scale, tag_info['width'], tag_info['angle'], theta_low, theta_high]


"""def get_theta(Width_px, t=0.7E-3, W=7.7E-3, scale=1):
    theta = (np.pi / 180) * np.linspace(0, 89.9, 1800)
    L = np.abs(W * np.cos(theta)) + np.abs(t * np.sin(theta))
    index = np.argmin(np.abs(scale * Width_px - L))
    theta_measured = theta[index] * 180 / np.pi
    return theta_measured
"""


def get_theta(Width_px, phi, t=0.7E-3, W=7.7E-3, scale=1):

    theta = (np.pi / 180) * np.linspace(0, 89.9, 1800)
    L = np.abs(W * np.cos(theta)) + np.abs(t * np.sin(theta))

    #plt.figure()
    #plt.plot(theta * 180 / np.pi, L)
    #plt.show()

    index_max = int(np.argmax(L))
    index_small = int(
        np.argmin(np.abs(scale * Width_px - L[:index_max] / np.cos(phi*np.pi/180))))
    index_large = int(np.argmin(np.abs(
        scale * Width_px - L[index_max:] / np.cos(phi*np.pi/180))))  # / np.cos(phi*np.pi/180)

    print('scale', scale)
    print('L', L)
    print('Width_px', Width_px)
    print('projected width', scale * Width_px)
    print('theta_small', theta[index_small] * 180 / np.pi)
    print('theta_max', theta[index_large + index_max] * 180 / np.pi)

    if index_small == 0:
        theta_measured_low = np.nan
    else:
        theta_measured_low = theta[index_small] * 180 / np.pi
    theta_measured_high = theta[index_large + index_max] * 180 / np.pi
    return theta_measured_low, theta_measured_high


if __name__ == '__main__':

    # pathname = 'Z:\\GranularCharge\\WhiteBead\\2023_11_06\\'
    pathname = 'E:/RawData/Mike/charge_papers_data/dipole_torque/2024_02_16/'
    #pathname = 'C:/Users/mikei/OneDrive - The University of Nottingham/Documents/Papers/Charge/Static_Charging/Figures/Figure2/2024_01_26/'
    filename = 'img*.png'

    # Reads the sequence
    readVid = ReadVideo(pathname + filename)
    img = readVid.read_frame(n=0)

    # Calc scale from diameter of ball
    scale, diam_px, height_px = get_scale(img)
    readVid.set_frame(n=0)

    # Processing params
    params = {'mask_top': 50, 'th1': 70,
              'scale': scale, 'width_bead': diam_px, 'configure': False}#threshold normally 100

    print(readVid.num_frames)
    # Setup dataframe to receive data
    df = pd.DataFrame(columns=['beadx', 'beady', 'beadx_m', 'beady_m', 'tag_proj_width',
                      'tag_angle_vertical', 'tag_rotation_angle_low', 'tag_rotation_angle_high'], index=range(1, readVid.num_frames + 1, 1))

    for i, img in enumerate(readVid):
        bead_and_tag = extract_bead_and_tag(img, **params)
        bead_info = find_bead_xy(bead_and_tag, diam_px)
        tag_info = find_tag(bead_and_tag, bead_info, diam_px)
        df.loc[i + 1] = get_data(tag_info, bead_info, scale)
        if i % 1 == 0:
            print(i)
            annotate_img(img.copy(), tag_info, bead_info, diam_px)

    df.to_csv(pathname + filename[:-5] + 'test.csv')
