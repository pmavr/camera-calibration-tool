import cv2
import numpy as np
from Camera import Camera
import utils


class TopViewer:

    def __init__(self, court_dimensions=(920, 592)):
        self.court_template = np.load('binary_court.npy')
        self.court_color = (0, 113, 0)
        self.left_side_color = (0, 0, 0)
        self.right_side_color = (0, 0, 0)
        self.desired_court_w = court_dimensions[0]
        self.desired_court_h = court_dimensions[1]
        court_w = int(Camera.court_length_x)
        court_h = int(Camera.court_width_y)
        self.f_w = court_w / self.desired_court_w
        self.f_h = court_h / self.desired_court_h

    def draw_court_top_view(self):
        edge_map = np.zeros((self.desired_court_h, self.desired_court_w, 3), dtype=np.uint8)
        n_line_segments = self.court_template.shape[0]
        homography = np.eye(3)

        for i in range(n_line_segments):
            line_seg = self.court_template[i]
            p1, p2 = line_seg[:2], line_seg[2:]

            q1 = Camera.project_point_on_topview(p1, homography, self.f_w, self.f_h)
            q2 = Camera.project_point_on_topview(p2, homography, self.f_w, self.f_h)

            q1 = Camera.point_to_int(q1)
            q2 = Camera.point_to_int(q2)
            cv2.line(edge_map, tuple(q1), tuple(q2), color=(255, 255, 255), thickness=2)

        line_seg = self.court_template[0]
        p1 = line_seg[:2]
        q1 = Camera.project_point_on_topview(p1, homography, self.f_w, self.f_h)
        q1 = Camera.point_to_int(q1)
        cv2.circle(edge_map, tuple(q1), radius=5, color=(255, 100, 255), thickness=2)

        return edge_map

    def project_field_of_view_on_top_view(self, h, color=(0, 255, 0)):
        court_top_view = self.draw_court_top_view()
        im = np.ones((720, 1280, 3), dtype=np.uint8)

        scale = np.array([[self.f_w, 0, 0],
                          [0, self.f_h, 0],
                          [0, 0, 1]])
        shift = np.array([[1, 0, 0],
                          [0, -1, Camera.court_width_y / self.f_h],
                          [0, 0, 1]])

        h = h @ scale
        h = h @ shift
        fov = cv2.warpPerspective(im, np.linalg.inv(h), (self.desired_court_w, self.desired_court_h),
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(0))
        fov = fov * np.array([[color]], dtype=np.uint8)  # green fov

        output = cv2.addWeighted(src1=court_top_view,
                                 src2=fov,
                                 alpha=.95, beta=.3, gamma=0.)
        return output