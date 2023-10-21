import numpy as np
import argparse
import cv2

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
            cv2.waitKey(0)

def main():
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

            image = capture_image(cameras[selected_camera][0])
            process_image(image)
        else:
            print("Invalid camera selection.")
    else:
        print("No cameras are available.")

if __name__ == "__main__":
    main()
