import cv2 as cv
import numpy as np
from pyciede2000 import ciede2000

import config
import functions as fn

# Set up camera and frame size
cam = cv.VideoCapture(0)
screen_resolution = 1280, 720
frame_x, frame_y = int(screen_resolution[0] / 4), int(screen_resolution[1] / 4)
frame_w, frame_h = frame_x + (2 * frame_x), frame_y + (2 * frame_y)

while True:
  
    # Get frame from webcam
    check, image = cam.read()
    
    # Crop the frame
    image = image[frame_y:frame_h, frame_x:frame_w]

    # Prepare the frame for contour detection
    grey_frame = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    noiseless_frame = cv.fastNlMeansDenoising(grey_frame, None, 20, 7, 7)
    blurred_frame = cv.blur(noiseless_frame, (3, 3))
    canny_frame = cv.Canny(blurred_frame, 30, 60, 3)
    dilated_frame = cv.dilate(canny_frame, cv.getStructuringElement(cv.MORPH_RECT, (9, 9)))

    # Detect the contours of the nine squares on the current face the Rubik's cube
    contours, _ = cv.findContours(dilated_frame, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Stores contours with the right dimensions and colours
    square_contours = []

    for contour in contours:

        # Get a polygon approximation for the contour 
        approx = cv.approxPolyDP(contour, 0.1*cv.arcLength(contour, True), True)
        
        #  Check the contour has four sides
        if len(approx) == 4:

            # Get dimensions of the contour and calculate w/h ratio and area
            x, y, w, h = cv.boundingRect(approx)
            ratio = float(w) / h
            area = cv.contourArea(approx)

            # Check the contour meets the right dimensions
            if ratio >= 0.8 and ratio <= 1.2 and w >= 30 and w <= 80 and area >= 900:

                # Determine the dominant colour in the contour and convert to LAB format
                dominant_square_colour_bgr = fn.get_dominant_colour(image[y:y+h, x:x+w])
                dominant_square_colour_lab = fn.convert_bgr_to_lab(dominant_square_colour_bgr)
            
                # Compare the dominant colour with each item in dictionary and select best match
                colour_deltas = []
                for rubik_colour, rubik_bgr in config.BGR_COLOURS.items():

                    # Determine colour delta
                    rubik_lab = fn.convert_bgr_to_lab(rubik_bgr)
                    colour_delta = ciede2000(rubik_lab, dominant_square_colour_lab)
                    
                    # Record colour and closeness
                    colour_deltas.append({"colour": rubik_colour, "delta": colour_delta["delta_E_00"]})

                # Find colour that had the smallest delta
                best_colour_delta = min(colour_deltas, key=lambda item: item["delta"])
                
                # Check whether the contour colour is a Rubik colour (and not mistaken for another object)
                # The smaller the value, the closer the match; poor light makes it harder to get close matches
                if best_colour_delta["delta"] < 80:

                    # This square contour is a good enough match with a Rubik colour
                    square_contours.append({"x": x, "y": y, "w": w, "h": h, "colour": best_colour_delta["colour"]})

    # Proceed if we found nine square contours with the right dimensions and colours
    if len(square_contours) == 9:
    
        # We need to map the square contours into this sequence based on their x-y positions
        # 0 1 2
        # 3 4 5
        # 6 7 8

        # Sort on x-positions
        square_contours_sorted_x = sorted(square_contours, key=lambda item: item["x"])

        # Sort on y-positions - this will help deduce the three rows below
        square_contours_sorted_y = sorted(square_contours, key=lambda item: item["y"])

        # Define the top, middle and bottom rows
        sorted_rows = []
        for i in range(0, 9, 3):

            # These three items are in the same row
            unsorted_row = [square_contours_sorted_y[i], square_contours_sorted_y[i+1], square_contours_sorted_y[i+2]]
            
            # Sort on x-position and append
            sorted_rows.append(sorted(unsorted_row, key=lambda item: item["x"]))

        # Re-order the list of square contours using sequence above
        square_contours = sorted_rows[0] + sorted_rows[1] + sorted_rows[2]

        # Define the middle square so that we can use that as a reference point x, y, w, h
        middle_square = square_contours[4]

        # Find the square contours furthest to the left (x-min) and right (x-max)
        x_min = square_contours_sorted_x[0]
        x_max = square_contours_sorted_x[-1]

        # Find the square contours furthest to the top (y-min) and bottom (y-max)
        y_min = square_contours_sorted_y[0]
        y_max = square_contours_sorted_y[-1]

        # Need to avoid outlier contours that aren't on the Rubik's cube
        # Check the four contours at the extreme x-min, x-max, y-min, y-max are positioned close enough to the middle square
        gap_width = int(middle_square["w"] * 1.7)
        gap_height = int(middle_square["h"] * 1.7)

        if (middle_square["x"] - x_min["x"] <= gap_width and
            x_max["x"] - middle_square["x"] <= gap_width and
            middle_square["y"] - y_min["y"] <= gap_height and 
            y_max["y"] - middle_square["y"] <= gap_height):

            # Good result, all nine square contours are close enough to the middle square
            fn.draw_contours(image, square_contours)
            fn.draw_mini_cube(image, square_contours)

    # Draw video window
    cv.imshow("Rubik's Cube Face Detection by Exactful", image)
   
    # Handle quit key q
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()