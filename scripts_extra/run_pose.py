import mediapipe as mp
import numpy as np
import cv2
import json

BODY_MAPPING_PIPE = np.array(
    [
        0,
        [11, 12],  # Right Eye
        12, # Right Shoulder
        14, # Right Elbow
        16, # Right Wrist
        11, # Left Shoulder
        13, # Left Elbow
        15, # Left Wrist
        [23.24],  # Right Eye Outer
        24, # Right Hip
        26, # Right Knee
        28, # Right Ankle
        23, # Left Hip
        25, # Left Knee
        27, # Left Ankle
        5,  # Left Eye
        2,  # Left Eye Inner
        8,  # Right Eye Inner
        7,  # Left Eye Outer
        29, # Left Foot Index (approximation of Left Big Toe)
        31, # Left Foot Index (approximation of Left Big Toe)
        [27, 29],  # Left Foot
        30, # Right Foot Index (approximation of Right Big Toe)
        32, # Right Foot Index (approximation of Right Big Toe)
        [28, 30],  # Right Foot
    ]
)
mp_pose = mp.solutions.pose
# Setting up the Pose model for images.
pose_img = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=1)
# Setting up the Pose model for videos.
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5,
                        min_tracking_confidence=0.5, model_complexity=1)
# Initializing mediapipe drawing class to draw landmarks on specified image.
mp_drawing = mp.solutions.drawing_utils

# Read the image
image_path = '/home/fares_uman3d_com/Clothed_SMPLX/data/input_test/female-19.png'
image = cv2.imread(image_path)

# Convert the image from BGR to RGB
RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Get the dimensions of the image
height, width, _ = image.shape
# Perform the Pose Detection
results = pose_img.process(RGB_img)

# Convert landmarks to a list of (x, y, visibility) in pixel coordinates
landmarks = [
    (int(landmark.x * height), int(landmark.y * width), landmark.z, landmark.visibility)
    for landmark in results.pose_landmarks.landmark
]

# Convert to a numpy array and extract only x, y (pixels) and visibility
mediapipe_output = np.array(landmarks)[:, [0, 1, 3]]  # Selecting x, y, visibility

print(mediapipe_output[BODY_MAPPING_PIPE])
# Flatten the MediaPipe data into OpenPose's expected format
pose_keypoints_2d = mediapipe_output.flatten().tolist()

# Build the OpenPose-like JSON structure
openpose_json = {
    "version": 1.3,
    "people": [
        {
            "person_id": [-1],
            "pose_keypoints_2d": pose_keypoints_2d,
            "face_keypoints_2d": [],
            "hand_left_keypoints_2d": [],
            "hand_right_keypoints_2d": [],
            "pose_keypoints_3d": [],
            "face_keypoints_3d": [],
            "hand_left_keypoints_3d": [],
            "hand_right_keypoints_3d": []
        }
    ]
}

# Convert to JSON string
openpose_json_str = json.dumps(openpose_json, indent=4)

# Save or print the result
with open("/home/fares_uman3d_com/Clothed_SMPLX/data/input_test/emale-19._json.json", "w") as f:
    f.write(openpose_json_str)