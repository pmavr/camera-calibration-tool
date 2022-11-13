import numpy as np
import cv2
import math


class Camera:
    court_length_x = 105.000552
    court_width_y = 68.003928

    def __init__(self, camera_params):
        self.image_center_x = camera_params[0]
        self.image_center_y = camera_params[1]
        self.focal_length = camera_params[2]
        self.tilt_angle = camera_params[3]
        self.pan_angle = camera_params[4]
        self.roll_angle = camera_params[5]
        self.camera_center_x = camera_params[6]
        self.camera_center_y = camera_params[7]
        self.camera_center_z = camera_params[8]
        self.camera_center = camera_params[6:9]
        base_rotation = self.rotate_y_axis(0) @ self.rotate_z_axis(self.roll_angle) @ \
                        self.rotate_x_axis(-90)
        pan_tilt_rotation = self.pan_y_tilt_x(self.pan_angle, self.tilt_angle)
        rotation = pan_tilt_rotation @ base_rotation
        self.rot_vector, _ = cv2.Rodrigues(rotation)
        self.image_width = int(2 * self.image_center_x)
        self.image_height = int(2 * self.image_center_y)

    def calibration_matrix(self):
        return np.array([[self.focal_length, 0, self.image_center_x],
                         [0, self.focal_length, self.image_center_y],
                         [0, 0, 1]])

    def rotation_matrix(self):
        rotation, _ = cv2.Rodrigues(self.rot_vector)
        return rotation

    def homography(self):
        P = self.projection_matrix()
        h = P[:, [0, 1, 3]]
        return h

    def projection_matrix(self):
        P = np.eye(3, 4)
        P[:, 3] = -1 * self.camera_center
        K = self.calibration_matrix()
        R = self.rotation_matrix()
        return K @ R @ P

    @staticmethod
    def rotate_x_axis(angle):
        """
        rotate coordinate with X axis
        https://en.wikipedia.org/wiki/Rotation_matrix + transpose
        http://mathworld.wolfram.com/RotationMatrix.html
        :param angle: in degree
        :return:
        """
        angle = math.radians(angle)
        s = math.sin(angle)
        c = math.cos(angle)

        r = np.asarray([[1, 0, 0],
                        [0, c, -s],
                        [0, s, c]])
        r = np.transpose(r)
        return r

    @staticmethod
    def rotate_y_axis(angle):
        """
        rotate coordinate with X axis
        :param angle:
        :return:
        """
        angle = math.radians(angle)
        s = math.sin(angle)
        c = math.cos(angle)

        r = np.asarray([[c, 0, s],
                        [0, 1, 0],
                        [-s, 0, c]])
        r = np.transpose(r)
        return r

    @staticmethod
    def rotate_z_axis(angle):
        """
        :param angle:
        :return:
        """
        angle = math.radians(angle)
        s = math.sin(angle)
        c = math.cos(angle)

        r = np.asarray([[c, -s, 0],
                        [s, c, 0],
                        [0, 0, 1]])
        r = np.transpose(r)
        return r

    def pan_y_tilt_x(self, pan, tilt):
        """
        Rotation matrix of first pan, then tilt
        :param pan:
        :param tilt:
        :return:
        """
        r_tilt = self.rotate_x_axis(tilt)
        r_pan = self.rotate_y_axis(pan)
        m = r_tilt @ r_pan
        return m

    @staticmethod
    def project_point_on_topview(point, h, s_w=1, s_h=1):
        x, y = point
        w = 1.0
        p = np.zeros(3)
        p[0], p[1], p[2] = x, y, w

        m1 = np.array([[1, 0, 0],
                       [0, -1, 68],
                       [0, 0, 1]])
        scale = np.array([[s_w, 0, 0],
                          [0, s_h, 0],
                          [0, 0, 1]])
        homography_matrix = h @ m1
        homography_matrix = homography_matrix @ scale
        inverted_homography_matrix = np.linalg.inv(homography_matrix)
        q = inverted_homography_matrix @ p

        assert q[2] != 0.0
        projected_x = np.rint(q[0] / q[2]).astype(np.int)
        projected_y = np.rint(q[1] / q[2]).astype(np.int)
        return [projected_x, projected_y]

    @staticmethod
    def project_point_on_frame(x, y, h):
        p = np.array([x, y, 1.])
        q = h @ p

        assert q[2] != 0.0
        projected_x = np.rint(q[0] / q[2]).astype(np.int)
        projected_y = np.rint(q[1] / q[2]).astype(np.int)
        return [projected_x, projected_y]

    def to_edge_map(self, court_template):
        edge_map = self.draw_court(court_template, with_points=False)
        self.draw_court_corners(edge_map)
        self.draw_image_center(edge_map)
        self.draw_human_dummy(edge_map)
        self.print_dist_from_camera(edge_map)

        return edge_map

    def draw_court(self, court_template, with_points=False):
        edge_map = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        n_line_segments = court_template.shape[0]

        for i in range(n_line_segments):
            line_seg = court_template[i]
            p1, p2 = line_seg[:2], line_seg[2:]

            q1 = Camera.project_point_on_frame(p1[0], p1[1], self.homography())
            q2 = Camera.project_point_on_frame(p2[0], p2[1], self.homography())

            if self._is_off_image_point(q1) and self._is_off_image_point(q2):
                continue

            cv2.line(edge_map, tuple(q1), tuple(q2), color=(255, 255, 255), thickness=2)

            if with_points:
                cv2.circle(edge_map, tuple(q1), radius=1, color=(0, 0, 255), thickness=2)
                cv2.circle(edge_map, tuple(q2), radius=1, color=(0, 0, 255), thickness=2)

        return edge_map

    def draw_court_corners(self, edge_map, color=(0, 0, 255)):
        points = [
            [self.court_length_x, self.court_width_y],  # upper-right corner
            [self.court_length_x, 0.],  # lower-right corner
            [0., 0.],  # lower-left corner
            [0., self.court_width_y]  # upper-left corner
        ]

        for p in points:
            p = Camera.project_point_on_frame(p[0], p[1], self.homography())
            cv2.circle(edge_map, tuple(p), radius=3, color=color, thickness=2)

    def draw_image_center(self, edge_map, color=(0, 255, 0)):
        cv2.line(edge_map,
                (int(self.image_center_x - 10), int(self.image_center_y)),
                (int(self.image_center_x + 10), int(self.image_center_y)),
                color=color, thickness=2)
        cv2.line(edge_map,
                (int(self.image_center_x), int(self.image_center_y - 10)),
                (int(self.image_center_x), int(self.image_center_y + 10)),
                color=color, thickness=2)

    def draw_human_dummy(self, edge_map, color=(0, 150, 255)):
        human_feet, human_height = self.generate_human_dummy()
        human_feet = Camera.project_point_on_frame(human_feet[0], human_feet[1], self.homography())
        cv2.line(edge_map, tuple(human_feet), (human_feet[0], human_feet[1] - int(human_height)), color=(0, 150, 255),
                 thickness=2)
        cv2.circle(edge_map, (human_feet[0], human_feet[1] - int(human_height)),
                   radius=4, color=color, thickness=3)
        cv2.putText(edge_map, f'{human_height:.2f}px',
                    (human_feet[0] + 5, human_feet[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, color)

    def print_dist_from_camera(self, edge_map, font_color=(0, 255, 0)):
        cv2.putText(edge_map, f'd:{self.distance_from_camera():.2f}m',
                    (int(self.image_center_x + 5), int(self.image_center_y) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, font_color)

    def distance_from_camera(self):
        return self.camera_center_z / np.cos(np.radians(90 - self.tilt_angle)) * (-1)

    def generate_human_dummy(self, height=1.8):
        dist_y_from_camera = self.camera_center_z * np.tan(np.radians(90 - self.tilt_angle)) * (-1)
        human_feet_xloc = dist_y_from_camera * np.cos(np.radians(90 - self.pan_angle)) + self.camera_center_x
        human_feet_yloc = dist_y_from_camera * np.sin(np.radians(90 - self.pan_angle)) + self.camera_center_y
        apparent_height = self.focal_length * height / self.distance_from_camera()
        return (human_feet_xloc, human_feet_yloc), apparent_height

    def _is_off_image_point(self, point):
        x, y = point
        return x < 0 or y < 0 or x > self.image_width or y > self.image_height

    def orientation(self):
        homography = self.homography()
        upper_right_corner = \
            Camera.project_point_on_frame(self.court_length_x, self.court_width_y, homography)[0]
        upper_left_corner = Camera.project_point_on_frame(0., self.court_width_y, homography)[0]
        lower_right_corner = Camera.project_point_on_frame(self.court_length_x, 0., homography)[0]
        lower_left_corver = Camera.project_point_on_frame(0., 0., homography)[0]

        if self.image_center_x in range(lower_left_corver, upper_left_corner):
            return 1
        elif self.image_center_x in range(upper_left_corner, upper_right_corner):
            return 0
        elif self.image_center_x in range(upper_right_corner, lower_right_corner):
            return 2
        else:
            return -1


