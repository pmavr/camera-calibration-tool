import cv2
import numpy as np
from Camera import Camera
from TopViewer import TopViewer
from wand.image import Image
from datetime import datetime

import pickle
import sys

import utils

bar_range = 500
pan_bar_range = 1000

camera_samples = []
Q = 113
R = 114
W = 119
A = 97
S = 115
D = 100
F = 102
G = 103
H = 104
J = 106
K = 107
L = 108


def normalize_in_range(value, min, max, b_range):
    return (((value - 0) * (max - min)) / (b_range - 0)) + min

def normalize_xloc_in_range(xloc):
    if camera_loc == 'master':
        return normalize_in_range(xloc, 46.2, 57.2, bar_range)
    elif camera_loc == 'offside_left':
        return normalize_in_range(xloc, 6.2, 47.2, bar_range)
    elif camera_loc == 'offside_right':
        return normalize_in_range(xloc, 57., 100., bar_range)
    elif camera_loc == 'high_behind_left':
        return normalize_in_range(xloc, -185., 0., bar_range)
    elif camera_loc == 'high_behind_right':
        return normalize_in_range(xloc, 105., 190., bar_range)

def normalize_yloc_in_range(yloc):
    if camera_loc == 'master':
        return normalize_in_range(yloc, -156., -25., bar_range)
    elif camera_loc == 'offside_left':
        return normalize_in_range(yloc, -156., -25., bar_range)
    elif camera_loc == 'offside_right':
        return normalize_in_range(yloc, -156., -25., bar_range)
    elif camera_loc == 'high_behind_left':
        return normalize_in_range(yloc, 15., 53., bar_range)
    elif camera_loc == 'high_behind_right':
        return normalize_in_range(yloc, 15., 53., bar_range)

def normalize_zloc_in_range(zloc):
    return normalize_in_range(zloc, 10.1387, 30.01126, bar_range)

def normalize_tilt_in_range(tilt_angle):
    return normalize_in_range(tilt_angle, -25., 0., bar_range)

def normalize_pan_in_range(pan_angle):
    if camera_loc == 'master':
        return normalize_in_range(pan_angle, -70., 70., pan_bar_range)
    elif camera_loc == 'offside_left':
        return normalize_in_range(pan_angle, -70., 70., pan_bar_range)
    elif camera_loc == 'offside_right':
        return normalize_in_range(pan_angle, -70., 70., pan_bar_range)
    elif camera_loc == 'high_behind_left':
        return normalize_in_range(pan_angle, 20., 160, pan_bar_range)
    elif camera_loc == 'high_behind_right':
        return normalize_in_range(pan_angle, -160., -20., pan_bar_range)

def update_record_param_trackbar(r):
    record_params = r
    update_image(1)

def update_focal_length_trackbar(fl):
    camera.focal_length = normalize_in_range(fl, 1000, 15000, bar_range)
    update_image(1)

def update_tilt_angle_trackbar(tilt):
    camera.tilt_angle = normalize_tilt_in_range(tilt)
    update_image(1)

def update_pan_angle_trackbar(pan):
    camera.pan_angle = normalize_pan_in_range(pan)
    update_image(1)

def update_roll_angle_trackbar(roll):
    camera.roll_angle = normalize_in_range(roll, -90., 90., bar_range)
    update_image(1)

def update_xloc_trackbar(x_loc):
    camera.camera_center_x = normalize_xloc_in_range(x_loc)
    update_image(1)

def update_yloc_trackbar(y_loc):
    camera.camera_center_y = normalize_yloc_in_range(y_loc)
    update_image(1)

def update_zloc_trackbar(z_loc):
    camera.camera_center_z = normalize_zloc_in_range(z_loc)
    update_image(1)

def update_dist1_trackbar(d1):
    camera.distortion_param_1 = normalize_in_range(d1, -.4, .4, pan_bar_range)
    update_image(1)

def update_dist2_trackbar(d2):
    camera.distortion_param_2 = normalize_in_range(d2, -.4, .4, pan_bar_range)
    update_image(1)

def update_dist3_trackbar(d3):
    camera.distortion_param_3 = normalize_in_range(d3, -.4, .4, pan_bar_range)
    update_image(1)


def update_image(val):
    camera_params = np.array([
        camera.image_center_x, camera.image_center_y,
        camera.focal_length, camera.tilt_angle, camera.pan_angle, camera.roll_angle,
        camera.camera_center_x, camera.camera_center_y, camera.camera_center_z
    ])

    if record_params_trackbar_val == 1:
        camera_samples.append(camera_params)

    homography = camera.homography()
    edge_map = camera.to_edge_map(court_template)
    img = Image.from_array(edge_map)
    img.distort('barrel', (camera.distortion_param_1, camera.distortion_param_2, camera.distortion_param_3, 1.))
    edge_map = np.array(img)
    font_color = (0, 0, 255)

    # uncomment to calibrate image
    im = cv2.imread('images/pas/pas_offr2.jpg')
    # im = cv2.imread(f'images/world_cup_2014/test_grouped_matches/recife.jpg')
    # im = cv2.imread(f'{path}{num}.jpg')
    im = cv2.resize(im, (1280, 720))
    edge_map = cv2.resize(edge_map, (im.shape[1], im.shape[0]))
    edge_map = cv2.addWeighted(src1=im,
                                   src2=edge_map,
                                   alpha=.95, beta=1, gamma=0.)

    text = f"focal length: {round(camera.focal_length, 3)} \n" \
           f"cam_loc_X: {round(camera.camera_center_x, 3)} \n" \
           f"cam_loc_Y: {round(camera.camera_center_y, 3)} \n" \
           f"cam_loc_Z: {round(camera.camera_center_z, 3)} \n" \
           f"tilt: {round(camera.tilt_angle, 3):.3f} \n" \
           f"pan: {round(camera.pan_angle, 3):.3f} \n" \
           f"roll: {round(camera.roll_angle, 3):.3f} \n" \
           f"Dist. 1: {round(camera.distortion_param_1, 3):.3f} \n" \
           f"Cam. Orient.: {camera.orientation()} \n" \
           f"Min. Focal Len.: {camera.max_focal_length_to_include_midpoints():.1f}\n"
    y0, dy = 30, 20
    for i, line in enumerate(text.split('\n')):
        y = y0 + i * dy
        cv2.putText(edge_map, line, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, font_color)
    homography /= homography[2][2]
    cv2.putText(edge_map,
                f'{np.round(homography[0, 0], 3):<8} {np.round(homography[0, 1], 3):=10} {np.round(homography[0, 2], 3):>10}',
                (900, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, font_color)
    cv2.putText(edge_map,
                f'{np.round(homography[1, 0], 3):<8} {np.round(homography[1, 1], 3):=10} {np.round(homography[1, 2], 3):>10}',
                (900, 50), cv2.FONT_HERSHEY_SIMPLEX, .5, font_color)
    cv2.putText(edge_map,
                f'{np.round(homography[2, 0], 3):<8} {np.round(homography[2, 1], 3):=10} {np.round(homography[2, 2], 3):>10}',
                (900, 70), cv2.FONT_HERSHEY_SIMPLEX, .5, font_color)
    cv2.putText(edge_map, f'Samples:{len(camera_samples)}', (600, y0), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))

    # top_viewer = TopViewer()
    # top_view = top_viewer.project_field_of_view_on_top_view(homography, color=(0, 255, 0))
    # top_view = cv2.resize(top_view, (960, 540))
    # tmp = np.zeros((106, 960, 3), dtype=np.uint8)
    # divider = np.ones((2, 960, 3), dtype=np.uint8) * 150
    # top_view = np.concatenate([top_view, divider, tmp], axis=0)
    # edge_map = cv2.resize(edge_map, (1152, 648))
    # divider = np.ones((648, 1, 3), dtype=np.uint8) * 150
    # full = np.concatenate([edge_map, divider, top_view], axis=1)
    full = edge_map
    cv2.imshow(title_window, full)


def save_camera_params(cam, file_path, file_num):
    cparams = str(cam)
    filename = f'{file_path}{file_num}.cparams'
    with open(filename, "w") as file:
        file.write(cparams)
    file.close()


if __name__ == '__main__':
    path = 'images/world_cup_2014/train_val/recife/'
    num = 133
    # camera_loc_x, camera_loc_y, camera_loc_z = 282, 432, 68 # belo horizonte
    # camera_loc_x, camera_loc_y, camera_loc_z = 285, 472, 97 # brazilia
    # camera_loc_x, camera_loc_y, camera_loc_z = 294, 454, 171 #285, 454, 153 #  recife
    # camera_loc_x, camera_loc_y, camera_loc_z = 269, 353, 204 # 263, 354, 190  # rio de janeiro
    # camera_loc_x, camera_loc_y, camera_loc_z = 286, 419, 228  # salvador
    # camera_loc_x, camera_loc_y, camera_loc_z = 300, 446, 105 # sao paulo
    # camera_loc_x, camera_loc_y, camera_loc_z = 283, 388, 259 # fortaleza
    # camera_loc_x, camera_loc_y, camera_loc_z = 300, 455, 82 # manaus


    camera_loc_x, camera_loc_y, camera_loc_z = 219, 464, 36 # 219, 470, 19 #  zosimades master
    camera_loc_x, camera_loc_y, camera_loc_z = 219, 464, 36  # zosimades offr
    camera_loc_x, camera_loc_y, camera_loc_z = 219, 470, 19  # zosimades offl
    # camera_loc_x, camera_loc_y, camera_loc_z = 285, 428, 61  # oaka

    court_template = utils.get_court_template()
    image_resolution = (1280, 720)
    image_center_x = image_resolution[0] * .5
    image_center_y = image_resolution[1] * .5
    focal_point_trackbar_val = int(pan_bar_range * .125)
    tilt_angle_trackbar_val = int(bar_range * .5)
    pan_angle_trackbar_val = int(pan_bar_range * .5)
    roll_angle_trackbar_val = int(bar_range * .5)
    camera_loc_x_trackbar_val = camera_loc_x # int(bar_range * .5)
    camera_loc_y_trackbar_val = camera_loc_y # int(bar_range * .5)
    camera_loc_z_trackbar_val = camera_loc_z# int(bar_range * .5)
    dis_1_trackbar_val = int(pan_bar_range * .5)
    # dis_2_trackbar_val = int(pan_bar_range * .5)
    # dis_3_trackbar_val = int(pan_bar_range * .5)
    record_params_trackbar_val = 0

    camera_loc = 'offside_right'   # master, offiside_left, high_behind_left
    camera_params = np.array([image_center_x, image_center_y,
                              normalize_in_range(focal_point_trackbar_val, 1000, 15000, bar_range),
                              normalize_tilt_in_range(tilt_angle_trackbar_val),
                              normalize_pan_in_range(pan_angle_trackbar_val),
                              normalize_in_range(roll_angle_trackbar_val, -90., 90., bar_range),
                              normalize_xloc_in_range(camera_loc_x_trackbar_val),
                              normalize_yloc_in_range(camera_loc_y_trackbar_val),
                              normalize_zloc_in_range(camera_loc_z_trackbar_val)
                              ])

    camera = Camera(camera_params)

    title_window = 'Camera Tool'
    cv2.namedWindow(title_window)
    cv2.createTrackbar('Record params', title_window, record_params_trackbar_val, 1, update_record_param_trackbar)
    cv2.createTrackbar('Focal length', title_window, focal_point_trackbar_val, pan_bar_range, update_focal_length_trackbar)
    cv2.createTrackbar('Tilt angle', title_window, tilt_angle_trackbar_val, bar_range, update_tilt_angle_trackbar)
    cv2.createTrackbar('Pan angle', title_window, pan_angle_trackbar_val, pan_bar_range, update_pan_angle_trackbar)
    cv2.createTrackbar('Roll angle', title_window, roll_angle_trackbar_val, bar_range, update_roll_angle_trackbar)
    cv2.createTrackbar('Camera loc x', title_window, camera_loc_x_trackbar_val, bar_range, update_xloc_trackbar)
    cv2.createTrackbar('Camera loc y', title_window, camera_loc_y_trackbar_val, bar_range, update_yloc_trackbar)
    cv2.createTrackbar('Camera loc z', title_window, camera_loc_z_trackbar_val, bar_range, update_zloc_trackbar)
    cv2.createTrackbar('Distortion param. 1', title_window, dis_1_trackbar_val, pan_bar_range, update_dist1_trackbar)
    # cv2.createTrackbar('Distortion param. 2', title_window, dis_2_trackbar_val, pan_bar_range, update_dist2_trackbar)
    # cv2.createTrackbar('Distortion param. 3', title_window, dis_3_trackbar_val, pan_bar_range, update_dist3_trackbar)
    update_image(1)

    while 1:
        key = cv2.waitKey(0)
        if key == Q:  # quit
            break
        elif key == W:
            val = cv2.getTrackbarPos('Tilt angle', title_window)
            cv2.setTrackbarPos('Tilt angle', title_window, val + 1)
            update_image(1)
        elif key == S:
            val = cv2.getTrackbarPos('Tilt angle', title_window)
            cv2.setTrackbarPos('Tilt angle', title_window, val - 1)
            update_image(1)
        elif key == A:
            val = cv2.getTrackbarPos('Pan angle', title_window)
            cv2.setTrackbarPos('Pan angle', title_window, val - 1)
            update_image(1)
        elif key == D:
            val = cv2.getTrackbarPos('Pan angle', title_window)
            cv2.setTrackbarPos('Pan angle', title_window, val + 1)
            update_image(1)


    save_camera_params(camera, path, num)

    sys.exit()
