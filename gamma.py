"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from ex1_utils import LOAD_GRAY_SCALE
import cv2
import numpy as np


def on_track(x):
    print(x / 100)


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    # openCV read it in revers so instead 1 it will be 2 and same for 2
    if rep == LOAD_GRAY_SCALE:
        img = cv2.imread(img_path, 2)
    else:
        img = cv2.imread(img_path, 1)
    # make it between 0 to 1
    img = img / 255
    img_copy = img.copy()
    track_bar_name = 'Gamma'
    window_name = 'Gamma Correction'
    cv2.namedWindow(window_name)
    # multiply by 100 in order to work with integers
    cv2.createTrackbar(track_bar_name, window_name, 100, 200, on_track)

    while True:
        cv2.imshow(window_name, img)
        gamma_pos = cv2.getTrackbarPos(track_bar_name, window_name)
        # divide by 100
        gamma = gamma_pos / 100
        # making correction according to the gamma
        img = np.power(img_copy, gamma)
        cv2.waitKey(1)


def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)
    # gammaDisplay('testimg1.jpg', 2)
    # gammaDisplay('testimg2.jpg', 2)


if __name__ == '__main__':
    main()
