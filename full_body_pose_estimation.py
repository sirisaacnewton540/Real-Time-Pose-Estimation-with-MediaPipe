import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe hands, face, pose, and iris classes.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)  # For iris tracking
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Open the webcam for real-time video capture.
cap = cv2.VideoCapture(0)

# Define a function to draw the 3D hand skeleton.
def draw_3d_skeleton(image, landmarks, connections):
    h, w, _ = image.shape
    landmarks_3d = [(int(lm.x * w), int(lm.y * h), lm.z * w) for lm in landmarks.landmark]

    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]
        start_point = landmarks_3d[start_idx]
        end_point = landmarks_3d[end_idx]

        # Draw line connections
        cv2.line(image, start_point[:2], end_point[:2], (255, 0, 0), 2)

    for landmark in landmarks_3d:
        # Draw landmark points
        cv2.circle(image, landmark[:2], 5, (0, 255, 0), -1)

# Define a function to draw the 3D facial landmarks.
def draw_3d_face(image, landmarks):
    h, w, _ = image.shape
    landmarks_3d = [(int(lm.x * w), int(lm.y * h), lm.z * w) for lm in landmarks.landmark]

    # Draw the landmarks
    for landmark in landmarks_3d:
        # Draw landmark points
        cv2.circle(image, landmark[:2], 2, (0, 0, 255), -1)

# Define a function to draw the 3D body landmarks, excluding face and hand landmarks.
def draw_3d_body(image, landmarks, connections):
    h, w, _ = image.shape
    excluded_landmarks = set(range(468, 489)) | set(range(522, 543))  # Face and hands landmarks indices

    landmarks_3d = [(int(lm.x * w), int(lm.y * h), lm.z * w) for lm in landmarks.landmark]

    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]
        if start_idx not in excluded_landmarks and end_idx not in excluded_landmarks:
            start_point = landmarks_3d[start_idx]
            end_point = landmarks_3d[end_idx]

            # Draw line connections
            cv2.line(image, start_point[:2], end_point[:2], (255, 0, 255), 2)

    for idx, landmark in enumerate(landmarks_3d):
        if idx not in excluded_landmarks:
            # Draw landmark points
            cv2.circle(image, landmark[:2], 5, (255, 255, 0), -1)

# Define a function to draw the iris landmarks.
def draw_3d_iris(image, landmarks):
    h, w, _ = image.shape
    iris_landmarks_indices = [468, 469, 470, 471, 472, 473, 474, 475]  # Iris landmarks

    for idx in iris_landmarks_indices:
        landmark = landmarks.landmark[idx]
        x, y = int(landmark.x * w), int(landmark.y * h)
        # Draw iris landmark points
        cv2.circle(image, (x, y), 2, (255, 0, 0), -1)

def recognize_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]

    thumb_ip = hand_landmarks.landmark[3]
    index_dip = hand_landmarks.landmark[6]
    middle_dip = hand_landmarks.landmark[10]
    ring_dip = hand_landmarks.landmark[14]
    pinky_dip = hand_landmarks.landmark[18]

    # Simple gesture logic (e.g., thumbs up, victory sign)
    if thumb_tip.y < thumb_ip.y and index_tip.y < index_dip.y and middle_tip.y < middle_dip.y and ring_tip.y > ring_dip.y and pinky_tip.y > pinky_dip.y:
        return "Victory"
    elif thumb_tip.y < thumb_ip.y and index_tip.y < index_dip.y and middle_tip.y > middle_dip.y and ring_tip.y > ring_dip.y and pinky_tip.y > pinky_dip.y:
        return "Thumbs Up"
    else:
        return "Unknown"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Create a blank image with the same dimensions as the frame.
    blank_frame = np.zeros_like(frame)

    # Convert the BGR image to RGB.
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the RGB image to get hand, face, and body landmarks.
    hand_results = hands.process(image_rgb)
    face_results = face_mesh.process(image_rgb)
    pose_results = pose.process(image_rgb)

    # Draw hand landmarks on the blank image and count fingers.
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # Draw the 3D skeleton on the blank image.
            draw_3d_skeleton(blank_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Draw facial landmarks and iris on the blank image.
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            # Draw the 3D face landmarks on the blank image.
            draw_3d_face(blank_frame, face_landmarks)
            # Draw the iris landmarks on the blank image.
            draw_3d_iris(blank_frame, face_landmarks)
        

    # Draw body landmarks on the blank image.
    if pose_results.pose_landmarks:
        draw_3d_body(blank_frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    
    # Display the blank frame with the hand, face, body, and iris landmarks.
    cv2.imshow('Hand, Face, Body, and Iris Pose Estimation', blank_frame)
    
    # Exit the loop when 'q' is pressed.
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows.
cap.release()
cv2.destroyAllWindows()
