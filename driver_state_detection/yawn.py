import cv2
import dlib

# Load the pre-trained facial landmark predictor from dlib
predictor_path = 'predictor/shape_predictor_68_face_landmarks.dat'  # Path to the shape predictor file
predictor = dlib.shape_predictor(predictor_path)

# Define a function to calculate the aspect ratio of the mouth
def calculate_mouth_aspect_ratio(shape):
    # Calculate the distances between the corners of the mouth
    left_distance = shape[48][1] - shape[54][1]
    right_distance = shape[64][1] - shape[60][1]
    top_distance = (shape[51][0] - shape[62][0] + shape[52][0] - shape[61][0]) / 2

    # Calculate the mouth aspect ratio
    mouth_aspect_ratio = (left_distance + right_distance) / (2 * top_distance)
    return mouth_aspect_ratio

# Define a function to detect yawning
def detect_yawning(shape, mouth_aspect_ratio_threshold):
    mouth_aspect_ratio = calculate_mouth_aspect_ratio(shape)
    if mouth_aspect_ratio > mouth_aspect_ratio_threshold:
        return True
    else:
        return False

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

# Define the mouth aspect ratio threshold for yawning detection
mouth_aspect_ratio_threshold = 0.2  # Adjust this threshold as needed

while True:
    # Read a frame from the webcam
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    detector = dlib.get_frontal_face_detector()
    faces = detector(gray)

    for face in faces:
        # Detect facial landmarks
        landmarks = predictor(gray, face)

        # Convert the landmarks to a NumPy array
        shape = [[p.x, p.y] for p in landmarks.parts()]

        # Draw the landmarks on the frame
        for point in shape:
            cv2.circle(frame, (point[0], point[1]), 2, (0, 255, 0), -1)

        # Detect yawning
        if detect_yawning(shape, mouth_aspect_ratio_threshold):
            cv2.putText(frame, "Yawning", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Yawning Detection', frame)

    # Check for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
