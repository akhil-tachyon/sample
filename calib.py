# measuring the diametr of coin or circle of a live video 
#HD 1080p = 67.5


import cv2
import numpy as np

# Known diameter of the reference object in millimeters
known_diameter_mm = 43.3  # 1 inch = 25.4 mm

# Known diameter of the reference object in pixels
known_diameter_pixels = 100  # Adjust as needed based on your calibration

# Initialize the camera (you can specify the camera index or video file path)
cap = cv2.VideoCapture(2)  # Use the default camera (change to a file path if needed)

while True:
    # Capture a frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to grayscale for circle detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    canny = cv2.Canny(blurred, 10, 50)

    cv2.imshow('canny', canny)

    # Detect circles using the Hough Circle Transform
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=10, maxRadius=100)

    # If circles are detected, draw them and measure the diameter in mm
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            radius_pixels = circle[2]
            
            # Calculate the diameter in millimeters
            diameter_mm = (radius_pixels / known_diameter_pixels) * known_diameter_mm
            
            # Draw the circle and diameter text
            cv2.circle(frame, center, radius_pixels, (0, 255, 0), 2)
            cv2.putText(frame, f"Diameter: {diameter_mm:.2f} mm", (center[0] - 70, center[1] + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the live video with detected circles
    cv2.imshow('Circle Detection', frame)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
