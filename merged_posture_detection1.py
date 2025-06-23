import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

# Initialize MediaPipe components
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

# Load YOLOv8 model
model = YOLO("yolov8l-cls.pt")

# Define classes for writing materials
WRITING_MATERIALS = {
    44: 'pen',       # bottle (can detect pen-like objects)
    47: 'pencil',    # pen/pencil
    76: 'paper',     # paper/document
}

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180 else 360 - angle

def is_writing_position(hand_landmarks):
    if not hand_landmarks:
        return False

    thumb_tip = hand_landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

    thumb_index_dist = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y]))
    thumb_middle_dist = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([middle_tip.x, middle_tip.y]))

    return thumb_index_dist < 0.05 and thumb_middle_dist < 0.05

# Start webcam
cap = cv2.VideoCapture(0)

# Initialize all MediaPipe solutions
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
     mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB for MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process with MediaPipe
        pose_results = pose.process(image)
        face_results = face_mesh.process(image)
        hand_results = hands.process(image)

        # Run YOLO for object detection
        yolo_results = model(frame, stream=True)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Default stage
        stage = "Unknown"
        
        try:
            landmarks = pose_results.pose_landmarks.landmark
            face_landmarks = face_results.multi_face_landmarks[0].landmark if face_results.multi_face_landmarks else None
            hand_landmarks = hand_results.multi_hand_landmarks[0].landmark if hand_results.multi_hand_landmarks else None

            # Extract key points
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                   landmarks[mp_pose.PoseLandmark.NOSE.value].y]

            # Calculate angles
            lefthand_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            righthand_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            leftshould_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
            rightshould_angle = calculate_angle(right_hip, right_shoulder, right_elbow)

            # Process YOLO detections
            pen_detected = False
            for r in yolo_results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    if class_id == 77:  # Pen detection
                        pen_detected = True
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"Pen {confidence:.2f}", (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Check for Writing Position
            if (pen_detected or is_writing_position(hand_landmarks)) and nose[1] > 0.5:  # Head down
                stage = "Writing"

            # Check for Hand Raising
            elif ((lefthand_angle < 60 or righthand_angle < 60) and 
                  (leftshould_angle > 25 or rightshould_angle > 25) and 
                  nose[1] < 0.6):
                stage = "Raising Hand"

            # Check for Active Listening
            elif face_landmarks:
                left_eye_inner = face_landmarks[133]
                right_eye_inner = face_landmarks[362]
                left_eye_outer = face_landmarks[33]
                right_eye_outer = face_landmarks[263]

                gaze_ratio = (right_eye_inner.x - left_eye_inner.x) / (right_eye_outer.x - left_eye_outer.x)
                shoulder_diff = abs(left_shoulder[1] - right_shoulder[1])
                
                if (0.3 < gaze_ratio < 0.7 and 
                    shoulder_diff < 0.08 and 
                    0.3 < nose[1] < 0.7):
                    stage = "Active Listening"

            # Draw pose landmarks
            mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        except Exception as e:
            print(f"Error processing frame: {e}")

        # Display the stage
        cv2.rectangle(image, (480, 20), (630, 120), (19, 69, 139), -1)
        cv2.putText(image, 'STAGE', (490, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, stage, (490, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('Posture Detection', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
