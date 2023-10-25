import numpy as np
import argparse
import cv2
import image_slicer
from PIL import Image
import os

outputSize = 600  # Change if needed to the desired output size

def list_available_cameras():
    # Iterate through camera indices and check if a camera is available
    available_cameras = []
    for camera_index in range(0, 10):  # You can adjust the range as needed
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            camera_info = get_camera_info(cap)
            available_cameras.append((camera_index, camera_info))
            cap.release()

    return available_cameras

def get_camera_info(cap):
    # Retrieve camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    return f"Resolution: {width}x{height}, FPS: {fps}"

def capture_image(camera_index):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Camera {camera_index} is not available.")
        return None

    ret, frame = cap.read()
    cap.release()

    if ret:
        return frame
    else:
        print(f"Error capturing image from Camera {camera_index}.")
        return None

def process_image(image):
    if image is not None:
        # Get image dimensions
        height, width, _ = image.shape

        # ArUco marker detection
        arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        arucoParams = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

        # Calculate the marker centers
        cornerCoordinates = []
        for i in range(len(corners)):
            cornerCoordinates.append([
                int(np.mean(corners[i][0][:, 0])),
                int(np.mean(corners[i][0][:, 1]))
            ])

        if len(cornerCoordinates) != 4:
            print("Error: There were only detected", len(cornerCoordinates), 'markers')
        else:
            # Init distortion variables
            pts1 = np.float32(cornerCoordinates)
            pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

            # Distort image to square
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            output = cv2.warpPerspective(image, matrix, (width, height))

            # Resize to the desired output size
            output = cv2.resize(output, (outputSize, outputSize), interpolation=cv2.INTER_CUBIC)

            # Display the result
            cv2.imshow("Image", output)
            #cv2.waitKey(0)
            
            # Convert the NumPy array to a Pillow Image
            output_pillow = Image.fromarray(output)
            # Specify the coordinates of the zoomed-in region (left, upper, right, lower)
            zoomed_region = (20, 23, 570, 600)  # Adjust these values as needed
            # Crop the image to the specified region
            zoomed_image = output_pillow.crop(zoomed_region)
            # Convert the Pillow Image to a NumPy array
            zoomed_image_np = np.array(zoomed_image)
            # Save or display the zoomed-in image
            cv2.imwrite("images_of_each_square/zoomed_image.jpg", zoomed_image_np)

# Function to calculate average color
def calculate_average_color(image):
    # Define the rows and columns to extract
    odd_rows = [1, 3, 5, 7]  # Odd-numbered rows
    odd_cols = [1, 3, 5, 7]  # Odd-numbered columns
    even_rows = [2, 4, 6, 8]  # Even-numbered rows
    even_cols = [2, 4, 6, 8]  # Even-numbered columns
    # Resize the image for more efficient processing
    resized_image = cv2.resize(image, (50, 50))
    # Calculate the average color
    average_color = np.mean(resized_image, axis=(0, 1))
    return average_color

def main():

    '''
    cameras = list_available_cameras()

    if cameras:
        print("Available cameras:")
        for i, (camera_index, camera_info) in enumerate(cameras):
            print(f"{i + 1}: Camera {camera_index}: {camera_info}")

        selected_camera = int(input("Enter the number of the camera you want to use: ")) - 1

        if 0 <= selected_camera < len(cameras):
            cap = cv2.VideoCapture(cameras[selected_camera][0])

            if not cap.isOpened():
                print(f"Error: Camera {cameras[selected_camera]} is not available.")
                return

            while True:
                ret, frame = cap.read()
                cv2.imshow("Camera Feed", frame)

                if cv2.waitKey(1) & 0xFF == 13:  # Enter key
                    break

            cap.release()
            cv2.destroyAllWindows()
'''

    #image = capture_image(cameras[selected_camera][0])
    image=cv2.imread("./Images/teste_damas_verdes1.jpg")
    process_image(image)
    
    #Slice the image in 8x8
    image_slicer.slice("images_of_each_square/zoomed_image.jpg", 64)
    
    sliced_images_directory = 'images_of_each_square' #Directory containing the saved images
    color_data = [] #Initialize a list to store color data
    
    # Define the rows and columns to extract
    odd_rows = [1, 3, 5, 7]  # Odd-numbered rows
    odd_cols = [1, 3, 5, 7]  # Odd-numbered columns
    even_rows = [2, 4, 6, 8]  # Even-numbered rows
    even_cols = [2, 4, 6, 8]  # Even-numbered columns

    # Loop through all image files in the directory
    for filename in os.listdir(sliced_images_directory):
        if filename.endswith(".png"):
            # Load the image
            image_path = os.path.join(sliced_images_directory, filename)
            image = cv2.imread(image_path)
             # Get the row and column number from the filename
            parts = filename.split("_")
            row, col = int(parts[2]), int(parts[3].split(".")[0])
            
            # Check if the row and column are in the specified sets
            if (row in odd_rows and col in odd_cols) or (row in even_rows and col in even_cols):
                # Load the image
                image_path = os.path.join(sliced_images_directory, filename)
                image = cv2.imread(image_path)
                average_color = calculate_average_color(image) #Calculate the average color of the image
                color_data.append((filename, average_color)) #Store the color data

    # Display or analyze the color data
    for filename, average_color in color_data:
        print(f"Image: {filename}, Average Color: {average_color}")


'''        else:
            print("Invalid camera selection.")
    else:
        print("No cameras are available.")
'''

if __name__ == "__main__":
    main()
