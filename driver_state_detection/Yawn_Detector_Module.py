import dlib
import cv2
import math
import numpy as np
def euclidean_distance(point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return distance
class YawnDetector:
    def __init__(self, yawn_threshold, consecutive_frames):
        self.yawn_threshold = yawn_threshold
        self.consecutive_frames = consecutive_frames
        self.yawn_counter = 0

        # Load pre-trained shape predictor model
        self.shape_predictor = dlib.shape_predictor("predictor/shape_predictor_68_face_landmarks.dat")

        # Define indices for the mouth landmarks
        self.mouth_indices = list(range(48, 68))

    def show_eye_keypoints(self, color_frame, landmarks):
        """
        Shows eyes keypoints found in the face, drawing red circles in their position in the frame/image

        Parameters
        ----------
        color_frame: numpy array
            Frame/image in which the eyes keypoints are found
        landmarks: list
            List of 68 dlib keypoints of the face
        """

    
        self.keypoints = landmarks

        for n in range(48, 68):
            x = self.keypoints.part(n).x
            y = self.keypoints.part(n).y
            cv2.circle(color_frame, (x, y), 1, (0, 0, 255), -1)
        return

    def detect_yawn(mouth_aspect_ratio, yawn_threshold):
        if mouth_aspect_ratio >= yawn_threshold:
            return True
        else:
            return False


    def compute_mouth_aspect_ratio(landmarks):
        # Extract the coordinates of the mouth landmarks
        left_mouth = (landmarks.part(48).x, landmarks.part(48).y)
        right_mouth = (landmarks.part(54).x, landmarks.part(54).y)
        top_lip1 = (landmarks.part(50).x, landmarks.part(50).y)
        top_lip2 = (landmarks.part(51).x, landmarks.part(51).y)
        bottom_lip1 = (landmarks.part(58).x, landmarks.part(58).y)
        bottom_lip2 = (landmarks.part(59).x, landmarks.part(59).y)

        # Calculate the distances between the mouth landmarks
        upper_distance = euclidean_distance(top_lip1, top_lip2)
        lower_distance = euclidean_distance(bottom_lip1, bottom_lip2)

        # Compute the mouth aspect ratio
        mouth_aspect_ratio = lower_distance / upper_distance

        return mouth_aspect_ratio




    # def compute_mouth_aspect_ratio(landmarks):
    #     # compute the euclidean distances between the vertical mouth landmarks
    #     A = np.linalg.norm(np.array([landmarks.part(3).x, landmarks.part(3).y]) - np.array([landmarks.part(9).x, landmarks.part(9).y]))
    #     B = np.linalg.norm(np.array([landmarks.part(2).x, landmarks.part(2).y]) - np.array([landmarks.part(10).x, landmarks.part(10).y]))
    #     C = np.linalg.norm(np.array([landmarks.part(4).x, landmarks.part(4).y]) - np.array([landmarks.part(8).x, landmarks.part(8).y]))

    #     # compute the euclidean distance between the horizontal mouth landmarks
    #     D = np.linalg.norm(np.array([landmarks.part(0).x, landmarks.part(0).y]) - np.array([landmarks.part(6).x, landmarks.part(6).y]))

    #     # compute the mouth aspect ratio
    #     mouth_aspect_ratio = (A + B + C) / (3 * D)
    #     return mouth_aspect_ratio




    # def compute_mouth_aspect_ratio(landmarks):
    # # compute the euclidean distances between the vertical mouth landmarks

    #     A = np.linalg.norm(landmarks.part(3) - landmarks.part(9))
    #     B = np.linalg.norm(landmarks.part(2) - landmarks.part(10))
    #     C = np.linalg.norm(landmarks.part(4) - landmarks.part(8))

    #     # compute the euclidean distance between the horizontal mouth landmarks
    #     D = np.linalg.norm(landmarks.part(0) - landmarks.part(6))

    #     # compute the mouth aspect ratio
    #     mouth_aspect_ratio = (Q + B + C) / (3 * D)

    #     return mouth_aspect_ratio

    # def compute_mouth_aspect_ratio(landmarks):
    #     # Compute distances between mouth landmarks
    #     left_dist = np.linalg.norm(landmarks[0] - landmarks[6])
    #     right_dist = np.linalg.norm(landmarks[3] - landmarks[9])
    #     top_dist = np.linalg.norm(landmarks[2] - landmarks[10])

    #     # Compute mouth aspect ratio
    #     mouth_aspect_ratio = (left_dist + right_dist) / (2 * top_dist)

    #     return mouth_aspect_ratio
