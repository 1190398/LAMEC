
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import json


# Function to detect and draw straight edges in an image
def detect_straight_edges_in_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # Perform Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 90)
    
    lines_coor = []

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            lines_coor.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })

        print("Straight edges detected")
        return lines_coor
        
    else:
        print("No straight edges detected")
        return []    

def detect_board_edges(lines, width, height):
    closest_lines = []

    for i in range(0, 360, 90):
        x1, y1 = width / 2, height / 2
        x2 = width / 2 + 2 * math.cos(i * math.pi / 180)
        y2 = height / 2 + 2 * math.sin(i * math.pi / 180)

        closest_line = None
        closest_distance = float("inf")

        for line in lines:
            x3, y3 = line["x1"], line["y1"]
            x4, y4 = line["x2"], line["y2"]

            den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

            if den == 0:
                continue

            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
            u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

            if t > 0 and u > 0 and u < 1:
                temp_x = x1 + t * (x2 - x1)
                temp_y = y1 + t * (y2 - y1)

                distance = math.sqrt((temp_x - width / 2) ** 2 + (temp_y - height / 2) ** 2)

                if distance < closest_distance:
                    closest_distance = distance
                    closest_line = {
                        "x1": int(x3),
                        "y1": int(y3),
                        "x2": int(x4),
                        "y2": int(y4)
                    }

        closest_lines.append(closest_line)

    return closest_lines

def find_intersections(closest_lines):
    intersections = []

    for i in range(4):
        line1 = closest_lines[i]
        for j in range(i + 1, 4):
            line2 = closest_lines[j]

            x1, y1, x2, y2 = line1["x1"], line1["y1"], line1["x2"], line1["y2"]
            x3, y3, x4, y4 = line2["x1"], line2["y1"], line2["x2"], line2["y2"]

            den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

            if den == 0:
                continue

            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
            u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

            if t > 0 and u > 0 and u < 1:
                temp_x = x1 + t * (x2 - x1)
                temp_y = y1 + t * (y2 - y1)
                intersections.append((int(temp_x), int(temp_y)))

    return intersections

# Function to capture an image from the webcam feed
def capture_image_from_webcam():
    cap = cv2.VideoCapture(1)

    ret, frame = cap.read()

    if not ret:
        print("Failed to capture a frame")
        cap.release()
        return None

    cap.release()
    return frame

def order_corners(corners, width, height):
    temp = []
    ordered = [None, None, None, None]

    for corner in corners:
        x, y = corner
        temp.append((x-width/2, y-height/2))

    for index, corner in enumerate(temp):
        x, y = corner
        x_f, y_f = corners[index]
        if x < 0 and y < 0: ordered[0] = (x_f, y_f)
        if x < 0 and y > 0: ordered[1] = (x_f, y_f)
        if x > 0 and y > 0: ordered[2] = (x_f, y_f)
        if x > 0 and y < 0: ordered[3] = (x_f, y_f)

    return ordered

def get_board(img, corners, margin):
    # Convert corners to NumPy array
    src_points = np.array(corners, dtype=np.float32)

    # Define the square points in the destination image to cover the entire output image
    dst_points = np.array([
        [margin, margin],
        [800 + margin - 1, margin],
        [800 + margin - 1, 800 + margin - 1],
        [margin, 800 + margin - 1]
    ], dtype=np.float32)
    # Find the homography matrix
    matrix, _ = cv2.findHomography(src_points, dst_points)

    # Apply the perspective transformation without cropping
    distorted_img = cv2.warpPerspective(img, matrix, (900, 900), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    distorted_img_resized = cv2.resize(distorted_img, (800, 800))

    rotated_image = cv2.rotate(distorted_img_resized, cv2.ROTATE_90_CLOCKWISE)

    return rotated_image

def divide_image_into_tiles(image, width, height, margin):
    tile_height = (height - 2 * margin) // 8
    tile_width = (width - 2 * margin) // 8

    newmargin = margin // 3

    tiles = []
    for i in range(8):
        for j in range(8):
            y_start = i * tile_height + margin - newmargin
            y_end = (i + 1) * tile_height + margin + newmargin
            x_start = j * tile_width + margin - newmargin
            x_end = (j + 1) * tile_width + margin + newmargin

            tile = image[y_start:y_end, x_start:x_end]

            tiles.append(tile)
    return tiles


def analyze_tile_for_round_object(tile):
    gray_tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(gray_tile, cv2.HOUGH_GRADIENT, dp=1, minDist=30, param1=40, param2=30, minRadius=28, maxRadius=40)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            cv2.circle(tile, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
            cv2.circle(tile, (circle[0], circle[1]), 2, (0, 0, 255), 3)

    return tile, len(circles) if circles is not None else 0, circles



def analyse_checker_color(tile, center, radius):
    # Ensure integer values for center and radius
    center = tuple(map(int, center))
    radius = int(radius)

    # Create a mask for the larger circular region
    mask_large = np.zeros_like(tile, dtype=np.uint8)
    cv2.circle(mask_large, center, radius, (255, 255, 255), thickness=cv2.FILLED)

    # Create a mask for the smaller circular region
    mask_small = np.zeros_like(tile, dtype=np.uint8)
    cv2.circle(mask_small, center, int(radius / 3), (255, 255, 255), thickness=cv2.FILLED)

    # Create a mask for the hole circular region
    mask_hole = np.zeros_like(tile, dtype=np.uint8)
    cv2.circle(mask_hole, center, int(radius / 5), (255, 255, 255), thickness=cv2.FILLED)


    # Resize the smaller circular mask to match the size of the larger one
    mask_outer_circular = cv2.resize(mask_small, (mask_large.shape[1], mask_large.shape[0]))

    # Subtract the smaller circular region from the larger one
    mask_circular_outer = cv2.subtract(mask_large, mask_outer_circular)

    # Extract the circular region without bitwise operations
    circular_region_outer = tile * (mask_circular_outer > 0)


    # Resize the smaller circular mask to match the size of the larger one
    mask_middle_circular = cv2.resize(mask_hole, (mask_small.shape[1], mask_small.shape[0]))

    # Subtract the smaller circular region from the larger one
    mask_circular_middle = cv2.subtract(mask_small, mask_middle_circular)

    # Extract the circular region without bitwise operations
    circular_region_middle = tile * (mask_circular_middle > 0)


    # Calculate the average color in the circular region
    total_color_circular_outer = np.sum(circular_region_outer, axis=(0, 1))
    total_pixels_circular_outer = np.sum(mask_circular_outer[:, :, 0] > 0)  # Count non-zero pixels in the mask

    # Calculate the average color in the circular region
    total_color_circular_middle = np.sum(circular_region_middle, axis=(0, 1))
    total_pixels_circular_middle = np.sum(mask_circular_middle[:, :, 0] > 0)  # Count non-zero pixels in the mask


    # Calculate the average color for circular region and central area
    average_color_circular_outer = total_color_circular_outer // total_pixels_circular_outer if total_pixels_circular_outer > 0 else (0, 0, 0)
    average_color_circular_middle = total_color_circular_middle // total_pixels_circular_middle if total_pixels_circular_middle > 0 else (0, 0, 0)


    
    return tuple(average_color_circular_outer), tuple(average_color_circular_middle)


def rgb_to_grayscale(rgb):
    # Convert RGB tuple to grayscale using a weighted average
    grayscale_value = int(0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])
    return grayscale_value


def subtract_images(color_image):

    # Read the image to be subtracted
    subtract_image = cv2.imread('steps/init/init_image.png')

    # Check if the images are of the same size
    if color_image.shape != subtract_image.shape:
        raise ValueError("Input images must have the same dimensions.")

    # Subtract the second image from the first
    result = cv2.subtract(color_image, subtract_image)

    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    #result = cv2.convertScaleAbs(result, alpha=5, beta=0)

    # Display the result (optional)
    cv2.imshow("Subtraction Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return result

def save_image(image):
    # Save the image
    cv2.imwrite('steps/init/init_image.png', image)

def read_corners_from_file():
    try:
        with open('steps/init/corners.json', 'r') as file:
            ordered_corners = json.load(file)
            return ordered_corners
    except Exception as e:
        print(f"An error occurred while reading corners from file: {e}")
        return None
    
# Main function
def main_recon():
    try:
        # Capture an image from the webcam
        captured_image = capture_image_from_webcam()

        if captured_image is not None:

            # Order corners based on their positions
            ordered_corners = read_corners_from_file()

            margin = 50

            # Get a warped image of the checkers board
            image = get_board(captured_image, ordered_corners, margin)

            # Get dimensions of the warped image
            width, height, channels = image.shape

            #sub_image = subtract_images(image)

            # Divide the image into tiles
            tiles = divide_image_into_tiles(image, width, height, margin)

            for i, tile in enumerate(tiles):
                try:
                    # Convert the image data type to uint8 if needed
                    imageshow = np.array(tile)
                    cv2.imwrite('steps/output/' + str(i) + '.png', imageshow)

                    # Initialize a matrix to represent the checkers on the board
                    checker_matrix = np.zeros((8, 8), dtype=int)

                    # Analyze each tile for a round object (checker)
                    for i, tile in enumerate(tiles):
                        try:
                            result_tile, object_count, objects = analyze_tile_for_round_object(tile)

                            if object_count > 0:
                                # Extract information about the detected checker
                                circle = objects[0, 0]
                                x, y, radius = circle

                                # Analyze the color of the checker
                                outer_color, middle_color = analyse_checker_color(tile, (x, y), radius * 0.9)

                                # Convert RGB to grayscale
                                outer_gray = rgb_to_grayscale(outer_color)
                                middle_gray = rgb_to_grayscale(middle_color)

                                print(str(outer_gray) + ' . ' + str(middle_gray))

                                if abs(outer_gray - middle_gray) < 40:  # not a queen
                                    # Update the checker matrix based on the color
                                    if outer_gray < 127:
                                        checker_matrix[i % 8][i // 8] = 1
                                    else:
                                        checker_matrix[i % 8][i // 8] = 2
                                else:  # its a queen
                                    # Update the checker matrix based on the color
                                    if outer_gray < 127:
                                        checker_matrix[i % 8][i // 8] = 3
                                    else:
                                        checker_matrix[i % 8][i // 8] = 4

                        except Exception as tile_analysis_exception:
                            print(f"Error analyzing tile {i}: {tile_analysis_exception}")

                    print(checker_matrix)

                    return checker_matrix

                except Exception as tile_save_exception:
                    print(f"Error saving tile {i}: {tile_save_exception}")

        else:
            print("Image capture failed")

    except Exception as main_exception:
        print(f"An error occurred in the main_recon function: {main_exception}")

def save_corners_to_file(ordered_corners):
    with open('steps/init/corners.json', 'w') as file:
        json.dump(ordered_corners, file)

# Function to capture an image from the webcam feed
def setup_web_cam():
    try:
        cap = cv2.VideoCapture(1)

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Failed to capture a frame")
                break

            # Capture an image when the Enter key is pressed
            key = cv2.waitKey(1)
            if key == 13:  # 13 corresponds to the Enter key

                # Detect straight edges in the captured image
                lines = detect_straight_edges_in_image(frame)
                width, height, channels = frame.shape

                # Detect closest lines to the edges of the board
                closest_lines = detect_board_edges(lines, width, height)

                # Find intersection points of the closest lines
                corners = find_intersections(closest_lines)

                # Order corners based on their positions
                ordered_corners = order_corners(corners, width, height)

                margin = 50

                # Get a warped image of the checkers board
                image = get_board(frame, ordered_corners, margin)

                save_image(image)

                # Save the ordered corner coordinates to a file
                save_corners_to_file(ordered_corners)


                cap.release()
                cv2.destroyAllWindows()
                return

            width, height, channels = frame.shape

            lines = detect_straight_edges_in_image(frame)
            closest_lines = detect_board_edges(lines, width, height)

            for line in closest_lines:
                if line is not None:
                    cv2.line(frame, (line["x1"], line["y1"]), (line["x2"], line["y2"]), (0, 255, 0), 1)

            cv2.circle(frame, (height // 2, width // 2), 2, (0, 255, 0), -1)

            cv2.imshow("Webcam Feed", frame)

            time.sleep(0.1)

        cap.release()
        cv2.destroyAllWindows()
        return None

    except Exception as webcam_setup_exception:
        print(f"An error occurred in the setup_web_cam function: {webcam_setup_exception}")

