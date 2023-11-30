
import cv2
import numpy as np
import math


# Function to detect and draw straight edges in an image
def detect_straight_edges_in_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # Perform Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 80)
    
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

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture a frame")
            break

        # Capture an image when the Enter key is pressed
        key = cv2.waitKey(1)
        if key == 13:  # 13 corresponds to the Enter key
            cap.release()
            cv2.destroyAllWindows()
            return frame
        
        width, height, channels = frame.shape
        
        lines = detect_straight_edges_in_image(frame)
        closest_lines = detect_board_edges(lines, width, height)

        for line in closest_lines:
            if line is not None:
                cv2.line(frame, (line["x1"], line["y1"]), (line["x2"], line["y2"]), (0, 255, 0), 1)

        cv2.circle(frame, (height//2, width//2), 2, (0, 255, 0), -1)


        cv2.imshow("Webcam Feed",frame)

    cap.release()
    cv2.destroyAllWindows()
    return None

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

def get_board(image, corners, width, height, margin):
    # Init distortion variables
    pts1 = np.float32(corners)
    pts2 = np.float32([[0, 0], [0, height], [width, height], [width, 0]])
        
    # Distort image to square with margin
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    image = cv2.warpPerspective(image, matrix, (width, height))

    # Resize to the desired output size
    image = cv2.resize(image, (800, 800), interpolation=cv2.INTER_CUBIC)

    avg_color = np.mean(image, axis=(0, 1))

    # Round the average color values to integers
    avg_color = tuple(np.round(avg_color).astype(int))

    blank_image = np.full((800+2*margin, 800+2*margin, 3), avg_color, dtype=np.uint8)


    # Ensure the overlay image is not larger than the base image
    overlay_height, overlay_width = image.shape[:2]
    base_height, base_width = blank_image.shape[:2]

    if overlay_height > base_height or overlay_width > base_width:
        print("Error: Overlay image is larger than the base image.")
        return None

    # Calculate the position to center the overlay image on the base image
    y_position = (base_height - overlay_height) // 2
    x_position = (base_width - overlay_width) // 2

    # Create a copy of the base image
    result = blank_image.copy()

    # Overlay the smaller image onto the larger image
    result[y_position:y_position + overlay_height, x_position:x_position + overlay_width] = image

    return result

def divide_image_into_tiles(image, width, height, margin):
    tile_height = (height - 2 * margin) // 8
    tile_width = (width - 2 * margin) // 8

    newmargin = margin // 2

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

    circles = cv2.HoughCircles(gray_tile, cv2.HOUGH_GRADIENT, dp=1, minDist=30, param1=40, param2=30, minRadius=5, maxRadius=50)

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

    # Create a mask for the circular region
    mask = np.zeros_like(tile)
    cv2.circle(mask, center, radius, (255, 255, 255), thickness=cv2.FILLED)

    # Use the mask to extract the circular region
    circular_region = cv2.bitwise_and(tile, mask)

    # Calculate the average color in the circular region
    total_color = np.sum(circular_region, axis=(0, 1))
    total_pixels = np.sum(mask[:, :, 0] > 0)  # Count non-zero pixels in the mask

    # Calculate the average color
    average_color = total_color // total_pixels if total_pixels > 0 else (0, 0, 0)

    return tuple(average_color)

def rgb_to_grayscale(color):
    value1, value2, value3 = color
    grayscale_value = 0.299 * value1 + 0.587 * value2 + 0.114 * value3
    return int(grayscale_value)



# Main function
def main_recon():
    # Capture an image from the webcam
    captured_image = capture_image_from_webcam()

    if captured_image is not None:
        # Detect straight edges in the captured image
        lines = detect_straight_edges_in_image(captured_image)
        width, height, channels = captured_image.shape

        # Detect closest lines to the edges of the board
        closest_lines = detect_board_edges(lines, width, height)

        # Find intersection points of the closest lines
        corners = find_intersections(closest_lines)

        # Order corners based on their positions
        ordered_corners = order_corners(corners, width, height)

        margin = 50

        # Get a warped image of the checkers board
        image = get_board(captured_image, ordered_corners, width, height, margin)

        # Display the warped image
        cv2.imshow("Webcam Feed", image)
        cv2.waitKey(0)

        # Get dimensions of the warped image
        width, height, channels = image.shape

        # Divide the image into tiles
        tiles = divide_image_into_tiles(image, width, height, margin)

        for i, tile in enumerate(tiles):
        # Convert the image data type to uint8 if needed
            imageshow = np.array(tile)
            cv2.imwrite('steps/output/' + str(i) + '.png', imageshow)

        # Initialize a matrix to represent the checkers on the board
        checker_matrix = np.zeros((8, 8), dtype=int)

        # Analyze each tile for a round object (checker)
        for i, tile in enumerate(tiles):

            result_tile, object_count, objects = analyze_tile_for_round_object(tile)

            if object_count > 0:
                # Extract information about the detected checker
                circle = objects[0, 0]
                x, y, radius = circle

                # Analyze the color of the checker
                color = rgb_to_grayscale(analyse_checker_color(tile, (x, y), radius * 0.9))

                print(color)

                # Update the checker matrix based on the color
                if color < 70:
                    checker_matrix[i // 8][i % 8] = 1
                else:
                    if color > 150:
                        checker_matrix[i // 8][i % 8] = 2

        print(checker_matrix)

        return checker_matrix
    else:
        print("Image capture failed")

main_recon()