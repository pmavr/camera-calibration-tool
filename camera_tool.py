import cv2
import numpy as np
from Camera import Camera
from TopViewer import TopViewer
from wand.image import Image
from datetime import datetime
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


def update_image(val):
    record_params = cv2.getTrackbarPos('Record params', title_window)
    fp = cv2.getTrackbarPos('Focal length', title_window)
    tilt_angle = cv2.getTrackbarPos('Tilt angle', title_window)
    pan_angle = cv2.getTrackbarPos('Pan angle', title_window)
    roll_angle = cv2.getTrackbarPos('Roll angle', title_window)
    xloc = cv2.getTrackbarPos('Camera loc x', title_window)
    yloc = cv2.getTrackbarPos('Camera loc y', title_window)
    zloc = cv2.getTrackbarPos('Camera loc z', title_window)
    dis_1 = cv2.getTrackbarPos('Distortion param. 1', title_window)
    dis_2 = cv2.getTrackbarPos('Distortion param. 2', title_window)
    dis_3 = cv2.getTrackbarPos('Distortion param. 3', title_window)

    fp = normalize_in_range(fp, 1000, 15000, bar_range)

    xloc = normalize_in_range(xloc, 46.2, 57.2, bar_range)
    yloc = normalize_in_range(yloc, -156., -25., bar_range)
    zloc = normalize_in_range(zloc, 10.1387, 30.01126, bar_range)

    tilt_angle = normalize_in_range(tilt_angle, -25., 0., bar_range)
    pan_angle = normalize_in_range(pan_angle, -70., 70., pan_bar_range)
    roll_angle = normalize_in_range(roll_angle, -90., 90., bar_range)

    dis_1 = normalize_in_range(dis_1, -.4, .4, pan_bar_range)
    dis_2 = normalize_in_range(dis_2, -.4, .4, pan_bar_range)
    dis_3 = normalize_in_range(dis_3, -.4, .4, pan_bar_range)

    params = np.array([
        image_center_x,
        image_center_y,
        fp,
        tilt_angle,
        pan_angle,
        roll_angle,
        xloc,
        yloc,
        zloc
    ])

    if record_params == 1:
        camera_samples.append(params)

    camera_params = np.array([
        image_center_x,
        image_center_y,
        fp,
        tilt_angle,
        pan_angle,
        roll_angle,
        xloc,
        yloc,
        zloc
    ])

    camera = Camera(camera_params)
    homography = camera.homography()
    edge_map = camera.to_edge_map(court_template)
    img = Image.from_array(edge_map)
    img.distort('barrel', (dis_1, dis_2, dis_3, 1.))
    edge_map = np.array(img)
    font_color = (0, 0, 255)

    # uncomment to calibrate image
    # im = cv2.imread('images/oaka_main.jpg')
    # im = cv2.imread('images/world_cup_2014/test_grouped_matches/brazilia/76.jpg')
    # im = cv2.resize(im, (1280, 720))
    # edge_map = cv2.resize(edge_map, (im.shape[1], im.shape[0]))
    # edge_map = cv2.addWeighted(src1=im,
    #                                src2=edge_map,
    #                                alpha=.95, beta=1, gamma=0.)

    text = f"focal length: {round(camera.focal_length, 3)} \n" \
           f"cam_loc_X: {round(camera.camera_center_x, 3)} \n" \
           f"cam_loc_Y: {round(camera.camera_center_y, 3)} \n" \
           f"cam_loc_Z: {round(camera.camera_center_z, 3)} \n" \
           f"tilt: {round(tilt_angle, 3):.3f} \n" \
           f"pan: {round(pan_angle, 3):.3f} \n" \
           f"roll: {round(roll_angle, 3):.3f} \n" \
           f"Dist. 1: {round(dis_1, 3):.3f} \n" \
           f"Dist. 2: {round(dis_2, 3):.3f} \n" \
           f"Dist. 3: {round(dis_3, 3):.3f} \n" \
           f"Cam. Orient.: {camera.orientation()} \n" \
           f"Min. Focal Len.: {camera.max_focal_length_to_include_midpoints():.1f}\n"
    y0, dy = 30, 20
    for i, line in enumerate(text.split('\n')):
        y = y0 + i * dy
        cv2.putText(edge_map, line, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, font_color)
    homography /= homography[2][2]
    cv2.putText(edge_map, f'{np.round(homography[0,0], 3):<8} {np.round(homography[0,1], 3):=10} {np.round(homography[0,2], 3):>10}', (900, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, font_color)
    cv2.putText(edge_map, f'{np.round(homography[1,0], 3):<8} {np.round(homography[1,1], 3):=10} {np.round(homography[1,2], 3):>10}', (900, 50), cv2.FONT_HERSHEY_SIMPLEX, .5, font_color)
    cv2.putText(edge_map, f'{np.round(homography[2,0], 3):<8} {np.round(homography[2,1], 3):=10} {np.round(homography[2,2], 3):>10}', (900, 70), cv2.FONT_HERSHEY_SIMPLEX, .5, font_color)
    cv2.putText(edge_map, f'Samples:{len(camera_samples)}', (600, y0), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))

    top_viewer = TopViewer()
    top_view = top_viewer.project_field_of_view_on_top_view(homography, color=(0, 255, 0))
    top_view = cv2.resize(top_view, (960, 540))
    tmp = np.zeros((106, 960, 3), dtype=np.uint8)
    divider = np.ones((2, 960, 3), dtype=np.uint8) * 150
    top_view = np.concatenate([top_view, divider, tmp], axis=0)
    edge_map = cv2.resize(edge_map, (1152, 648))
    divider = np.ones((648, 1, 3), dtype=np.uint8) * 150
    full = np.concatenate([edge_map, divider, top_view], axis=1)
    cv2.imshow(title_window, full)


def save_camera_samples():
    samples = np.array(camera_samples)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    num_of_samples = len(camera_samples)
    filename = f'saved_camera_param_data/{timestamp}-{num_of_samples}.npy'
    np.save(filename, samples)


if __name__ == '__main__':
    # camera_loc_x, camera_loc_y, camera_loc_z = 282, 432, 68 # belo horizonte
    # camera_loc_x, camera_loc_y, camera_loc_z = 285, 472, 97 # brazilia
    # camera_loc_x, camera_loc_y, camera_loc_z = 288, 447, 179 # recife
    # camera_loc_x, camera_loc_y, camera_loc_z = 269, 368, 175 # rio de janeiro
    # camera_loc_x, camera_loc_y, camera_loc_z = 286, 432, 194 # salvador
    # camera_loc_x, camera_loc_y, camera_loc_z = 299, 453, 88 # sao paulo
    camera_loc_x, camera_loc_y, camera_loc_z = 219, 470, 19 # zosimades

    court_template = np.load('binary_court.npy')
    image_resolution = (1280, 720)
    image_center_x = image_resolution[0] * .5
    image_center_y = image_resolution[1] * .5
    focal_point = int(pan_bar_range * .125)
    tilt_angle = int(bar_range * .65)
    pan_angle = int(pan_bar_range * .5)
    roll_angle = int(bar_range * .5)
    # camera_loc_x = int(bar_range * .57)
    # camera_loc_y = int(bar_range * .5)
    # camera_loc_z = int(bar_range * .5)
    dis_1 = int(pan_bar_range * .5)
    dis_2 = int(pan_bar_range * .5)
    dis_3 = int(pan_bar_range * .5)
    record_params = 0

    title_window = 'Camera Tool'
    cv2.namedWindow(title_window)
    cv2.createTrackbar('Record params', title_window, record_params, 1, update_image)
    cv2.createTrackbar('Focal length', title_window, focal_point, pan_bar_range, update_image)
    cv2.createTrackbar('Tilt angle', title_window, tilt_angle, bar_range, update_image)
    cv2.createTrackbar('Pan angle', title_window, pan_angle, pan_bar_range, update_image)
    cv2.createTrackbar('Roll angle', title_window, roll_angle, bar_range, update_image)
    cv2.createTrackbar('Camera loc x', title_window, camera_loc_x, bar_range, update_image)
    cv2.createTrackbar('Camera loc y', title_window, camera_loc_y, bar_range, update_image)
    cv2.createTrackbar('Camera loc z', title_window, camera_loc_z, bar_range, update_image)
    cv2.createTrackbar('Distortion param. 1', title_window, dis_1, pan_bar_range, update_image)
    cv2.createTrackbar('Distortion param. 2', title_window, dis_2, pan_bar_range, update_image)
    cv2.createTrackbar('Distortion param. 3', title_window, dis_3, pan_bar_range, update_image)
    update_image(1)

    while 1:
        key = cv2.waitKey(0)
        if key == Q:    # quit
            break
        elif key == W:
            val = cv2.getTrackbarPos('Tilt angle', title_window)
            cv2.setTrackbarPos('Tilt angle', title_window, val+1)
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

    if len(camera_samples) > 0:
        save_camera_samples()
        print('Camera samples saved!')
    sys.exit()
