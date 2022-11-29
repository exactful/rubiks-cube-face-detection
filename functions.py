import config
import cv2 as cv
import numpy as np

def draw_contours(image, squares):
    pos = 0
    for square in squares:
        x, y, w, h = square["x"], square["y"], square["w"], square["h"]
        cv.rectangle(image, (x, y), (x+w, y+h), (220, 220, 220), 5)
        cv.putText(image, f"{pos}", (x+int(0.5*w), y+int(0.5*h)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv.LINE_AA)
        pos += 1

def draw_mini_cube(image, squares):
    x, y = config.MINI_CUBE_X, config.MINI_CUBE_Y
    pos = 0
    for i in range(3):
        for j in range(3):
            cv.rectangle(image, (x, y), (x + config.MINI_CUBE_STICKER_W, y + config.MINI_CUBE_STICKER_H), config.BGR_COLOURS[squares[pos]["colour"]], -1)
            x += 25
            pos += 1
        x = config.MINI_CUBE_X
        y += 25

def get_dominant_colour(image):

    data = image.reshape(-1, 3)
    n_clusters = 1
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv.KMEANS_RANDOM_CENTERS

    _, labels, centers = cv.kmeans(data.astype(np.float32), n_clusters, None, criteria, 10, flags)
    
    return tuple(centers[0])

def convert_bgr_to_lab(bgr_colour) :

    # Lifted from https://stackoverflow.com/a/16020102

    rgb_colour = (bgr_colour[2], bgr_colour[1], bgr_colour[0])

    num = 0
    RGB = [0, 0, 0]

    for value in rgb_colour :
        value = float(value) / 255

        if value > 0.04045 :
            value = ((value + 0.055) / 1.055) ** 2.4
        else :
            value = value / 12.92

        RGB[num] = value * 100
        num = num + 1

    XYZ = [0, 0, 0,]

    X = RGB[0] * 0.4124 + RGB[1] * 0.3576 + RGB[2] * 0.1805
    Y = RGB[0] * 0.2126 + RGB[1] * 0.7152 + RGB[2] * 0.0722
    Z = RGB[0] * 0.0193 + RGB[1] * 0.1192 + RGB[2] * 0.9505

    XYZ[0] = round(X, 4)
    XYZ[1] = round(Y, 4)
    XYZ[2] = round(Z, 4)

    XYZ[0] = float(XYZ[0]) / 95.047         # ref_X =  95.047   Observer= 2°, Illuminant= D65
    XYZ[1] = float(XYZ[1]) / 100.0          # ref_Y = 100.000
    XYZ[2] = float(XYZ[2]) / 108.883        # ref_Z = 108.883

    num = 0
    for value in XYZ :

        if value > 0.008856 :
            value = value ** (0.3333333333333333)
        else :
            value = (7.787 * value) + (16 / 116)

        XYZ[num] = value
        num = num + 1

    Lab = [0, 0, 0]

    L = (116 * XYZ[1]) - 16
    a = 500 * (XYZ[0] - XYZ[1])
    b = 200 * (XYZ[1] - XYZ[2])

    Lab [0] = round(L, 4)
    Lab [1] = round(a, 4)
    Lab [2] = round(b, 4)

    return Lab