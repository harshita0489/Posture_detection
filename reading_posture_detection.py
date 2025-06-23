import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

# Initialize MediaPipe components
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Load YOLOv8 model
model = YOLO("yolov8l.pt")

# Define classes that could be reading materials
READING_MATERIALS = {
    73: 'book',  # book
    74: 'notebook',  # notebook
    76: 'paper',  # paper/document
    84: 'book_alt'  # alternative book class
}

def calculate_head_pose(face_landmarks):
    """Calculate head pose (pitch, yaw, roll) using face landmarks"""
    nose = np.array([face_landmarks.landmark[1].x, face_landmarks.landmark[1].y, face_landmarks.landmark[1].z])
    chin = np.array([face_landmarks.landmark[152].x, face_landmarks.landmark[152].y, face_landmarks.landmark[152].z])
    pitch = np.degrees(np.arctan2(chin[1] - nose[1], abs(chin[2] - nose[2] + 1e-6)))
    return pitch

def is_reading_position(face_landmarks, book_box):
    """Determine if the user is in a reading position"""
    # Get face center
    nose_point = face_landmarks.landmark[1]
    face_center = np.array([nose_point.x, nose_point.y])

    # Check head tilt
    pitch = calculate_head_pose(face_landmarks)
    head_tilt_ok = 10 < pitch < 60

    # Check book position
    if book_box and head_tilt_ok:
        x1, y1, x2, y2 = book_box
        book_center_x = (x1 + x2) / 2 / 640
        book_center_y = (y1 + y2) / 2 / 480
        
        book_below_face = book_center_y > face_center[1]
        horizontal_distance = abs(book_center_x - face_center[0])
        vertical_distance = book_center_y - face_center[1]
        
        return (book_below_face and 
                horizontal_distance < 0.35 and 
                vertical_distance < 0.6)
    return False

# Start webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe solutions
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        face_results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Process YOLO detections
        book_box = None
        yolo_results = model(frame, stream=True)
        for r in yolo_results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                if class_id in READING_MATERIALS and confidence > 0.35:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    book_box = (x1, y1, x2, y2)
                    break

        # Check reading position and display status
        if face_results.multi_face_landmarks and book_box:
            face_landmarks = face_results.multi_face_landmarks[0]
            is_reading = is_reading_position(face_landmarks, book_box)
            
            # Display reading status
            status_text = "READING" if is_reading else "NOT READING"
            color = (0, 255, 0) if is_reading else (0, 0, 255)
            
            # Add background to text for better visibility
            text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            cv2.rectangle(image, (10, 30), (text_size[0] + 20, 70), (0, 0, 0), -1)
            cv2.putText(image, status_text, (15, 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        cv2.imshow('Reading Posture Detection', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
