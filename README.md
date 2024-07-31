# Real-Time Pose Estimation with MediaPipe

## Introduction

This project utilizes MediaPipe to capture and display real-time 3D hand, face, body, and iris landmarks using a webcam. It provides functionalities to recognize basic hand gestures and visualize skeletal structures.

## Results
![GIFMaker_pose](https://github.com/user-attachments/assets/93f31a7e-a2bb-4cfb-b06f-d42d765d5bfd)

![GIFMaker_full_pose](https://github.com/user-attachments/assets/db874598-1317-4669-bcd3-5caffdd18366)

## Libraries Used

### 1. OpenCV

- **Open Source Computer Vision Library**: OpenCV is an open-source library that provides a common infrastructure for computer vision applications. It includes hundreds of computer vision algorithms that can be used for image processing, video capture, and analysis.
- **Usage in this project**: OpenCV is used to capture video from the webcam, process image frames, and display the processed frames with overlaid landmarks.

### 2. MediaPipe

- **Cross-platform Framework**: MediaPipe is a cross-platform framework for building multimodal applied machine learning pipelines. It is developed by Google and includes ready-to-use solutions for face detection, hand tracking, pose estimation, object detection, and more.
- **Usage in this project**: MediaPipe is used to detect and track the 3D landmarks of hands, face, body, and iris in real-time.

### 3. NumPy

- **Numerical Computing Library**: NumPy is a powerful library for numerical computing in Python. It provides support for arrays, matrices, and many mathematical functions to operate on these data structures.
- **Usage in this project**: NumPy is used to create blank frames for drawing the landmarks and for efficient numerical operations on image data.

## Code Explanation

### Initializing MediaPipe Solutions

```python
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
```

- **mp_hands, mp_face_mesh, mp_pose**: These are MediaPipe solutions for hand tracking, face mesh, and pose estimation, respectively.
- **hands, face_mesh, pose**: These are instances of the respective MediaPipe solutions, configured to detect and track landmarks.

### Opening the Webcam

```python
# Open the webcam for real-time video capture.
cap = cv2.VideoCapture(0)
```

- **cap**: This is an OpenCV VideoCapture object that captures video from the default webcam (index 0).

### Drawing Functions

#### Draw 3D Hand Skeleton

```python
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
```

- **draw_3d_skeleton**: This function draws the 3D hand skeleton by connecting the landmarks with lines and drawing circles at each landmark position.

#### Draw 3D Facial Landmarks

```python
# Define a function to draw the 3D facial landmarks.
def draw_3d_face(image, landmarks):
    h, w, _ = image.shape
    landmarks_3d = [(int(lm.x * w), int(lm.y * h), lm.z * w) for lm in landmarks.landmark]

    # Draw the landmarks
    for landmark in landmarks_3d:
        # Draw landmark points
        cv2.circle(image, landmark[:2], 2, (0, 0, 255), -1)
```

- **draw_3d_face**: This function draws the 3D facial landmarks by placing circles at each landmark position.

#### Draw 3D Body Landmarks

```python
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
```

- **draw_3d_body**: This function draws the 3D body landmarks by connecting the landmarks with lines and drawing circles at each landmark position, excluding the face and hand landmarks.

#### Draw 3D Iris Landmarks

```python
# Define a function to draw the iris landmarks.
def draw_3d_iris(image, landmarks):
    h, w, _ = image.shape
    iris_landmarks_indices = [468, 469, 470, 471, 472, 473, 474, 475]  # Iris landmarks

    for idx in iris_landmarks_indices:
        landmark = landmarks.landmark[idx]
        x, y = int(landmark.x * w), int(landmark.y * h)
        # Draw iris landmark points
        cv2.circle(image, (x, y), 2, (255, 0, 0), -1)
```

- **draw_3d_iris**: This function draws the iris landmarks by placing circles at the specified landmark positions.

### Main Loop

```python
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
```

- **Main Loop**: Captures frames from the webcam, processes the frames to detect and draw landmarks, and displays the output in a separate window.
  - **cap.read()**: Captures a frame from the webcam.
  - **np.zeros_like(frame)**: Creates a blank frame with the same dimensions as the captured frame.
  - **cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)**: Converts the BGR image to RGB for processing with MediaPipe.
  - **hands.process(image_rgb), face_mesh.process(image_rgb), pose.process(image_rgb)**: Processes the RGB image to detect and track hand, face, and body landmarks.
  - **draw_3d_skeleton, draw_3d_face, draw_3d_iris, draw_3d_body**: Draws the detected landmarks on the blank frame.
  - **cv2.imshow**: Displays the processed frame with landmarks.
  - **cv2.waitKey(10)**: Waits for 10 milliseconds for a key press.
  - **cap.release(), cv2.destroyAllWindows()**: Releases the webcam and closes all OpenCV windows.

## Acknowledgements

This project uses the following libraries:
- [OpenCV](https://opencv.org/)
- [MediaPipe](https://mediapipe.dev/)
- [NumPy](https://numpy.org/)
