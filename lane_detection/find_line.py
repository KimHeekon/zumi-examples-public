from skimage.draw import line
import numpy as np
import math
import os
import sys
import matplotlib.pyplot as plt
import fire
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
if True:
    import cv2


calibration_matrix = np.array([[308.37228646, 0.,      163.13680689],
                               [0.,    308.91432819, 116.00877425],
                               [0.,    0.,     1.]])
calibration_matrix_inverse = np.linalg.inv(calibration_matrix)
KNOWN_HEIGHT = 0.019
HORIZONTAL_OFFSET = 80


def main(video=True):
    if video:
        cap = cv2.VideoCapture('samples/video/sample.avi')
        while(cap.isOpened()):
            ret, frame = cap.read()
            if frame is None:
                break
            frame = cv2.resize(frame, dsize=(160, 128))
            detect_lane(frame)
        cap.release()
    else:
        png_path = "samples/img/"
        png_list = os.listdir(png_path)
        png_list.sort()
        for png_name in png_list:
            img = cv2.imread(png_path + png_name)
            detect_lane(img)


def from_img_to_camera(x, y):
    x_, y_ = x * 320 / 160, (y + HORIZONTAL_OFFSET) * 320 / 128
    v = np.matmul(calibration_matrix_inverse, np.array([x_, y_, 1]))
    return [v[0]*KNOWN_HEIGHT/v[1], KNOWN_HEIGHT, v[2]*KNOWN_HEIGHT/v[1]]


def detect_lane(original_img):
    img = original_img[HORIZONTAL_OFFSET:]
    height, width, _ = img.shape
    img_canny = cv2.Canny(img, 100, 200)
    lines = cv2.HoughLines(img_canny, 1, np.pi/180, 30)
    dist_transform = cv2.distanceTransform(
        255-img_canny, cv2.DIST_L2, 5)
    if lines is None:
        return
    for i in range(len(lines)):
        for rho, theta in lines[i]:
            if abs(rho) < 10:
                continue
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = width - 1
            y1 = int(y0 + (width - 1 - x0) / (-b) * a)
            x2 = 0
            y2 = int(y0 - (-x0)*(a))
            x3 = int(x0 - (height - 1 - y0) / (-a) * (-b))
            y3 = height - 1
            x4 = int(x0 - (-y0) / (-a) * (-b))
            y4 = 0
            point_list = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
            filtered_point_list = []
            for point in point_list:
                if 0 <= point[0] < width and 0 <= point[1] < height:
                    filtered_point_list.append(point)
            filtered_point_num = len(filtered_point_list)
            if filtered_point_num > 2:
                def norm(x, y):
                    return (x[0] - y[0]) * (x[0] - y[0]) + (x[1] - y[1]) * (x[1] - y[1])
                max_dist_pts = 0, 0, 0
                for i in range(filtered_point_num - 1):
                    for j in range(filtered_point_num - 1 - i):
                        dist = norm(
                            filtered_point_list[i], filtered_point_list[i+j+1])
                        if max_dist_pts[0] < dist:
                            max_dist_pts = dist, i, i + j + 1
                filtered_point_list = [
                    filtered_point_list[max_dist_pts[1]], filtered_point_list[max_dist_pts[2]]]
            d_line = line(*filtered_point_list[0], *filtered_point_list[1])
            real_line = (dist_transform[d_line[1], d_line[0]] <= 1)
            real_line_points = d_line[0][real_line], d_line[1][real_line]
            # first_pt = from_img_to_camera(
            #     real_line_points[0][0], real_line_points[1][0])
            # end_pt = from_img_to_camera(
            #     real_line_points[0][-1], real_line_points[1][-1])
            # line_theta = np.arctan2(
            #     end_pt[2]-first_pt[2], end_pt[0]-first_pt[0])
            original_img[real_line_points[1]+80,
                         real_line_points[0]] = [255, 0, 0]
    cv2.imshow("img", original_img)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        exit()


if __name__ == "__main__":
    fire.Fire(main)
