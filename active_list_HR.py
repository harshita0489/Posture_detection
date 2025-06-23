import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180 else 360 - angle

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        pose_results = pose.process(image)
        face_results = face_mesh.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Default to "unknown"
        stage = "Unknown"
        
        try:
            landmarks = pose_results.pose_landmarks.landmark
            face_landmarks = face_results.multi_face_landmarks[0].landmark if face_results.multi_face_landmarks else None

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
            
            lefthand_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            righthand_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            leftshould_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
            rightshould_angle = calculate_angle(right_hip, right_shoulder, right_elbow)
           

            # Check for "Raising Hand" FIRST
            # if lefthand_angle < 80 or righthand_angle < 80:  # Loosen hand raise condition
            #     if nose[1] < 0.7:  # Ensures the person is facing forward
            #         stage = "Raising Hand"
            if ((lefthand_angle < 60 or righthand_angle < 60)  # At least one hand raised
                # and not (lefthand_angle < 60 and righthand_angle < 60)
                and (leftshould_angle > 25 or rightshould_angle > 25)):  # Shoulders confirm raise 
                if nose[1] < 0.6:  # Ensures user is facing forward
                    stage = "Raising hand"

            # If NOT Raising Hand, check for "Active Listening"
             # Eye landmarks for gaze detection
            elif face_landmarks: 
                left_eye_inner = face_landmarks[133]  # Inner left eye corner
                right_eye_inner = face_landmarks[362]  # Inner right eye corner
                left_eye_outer = face_landmarks[33]  # Outer left eye corner
                right_eye_outer = face_landmarks[263]  # Outer right eye corner

                # Calculate horizontal gaze direction
                gaze_ratio = (right_eye_inner.x - left_eye_inner.x) / (right_eye_outer.x - left_eye_outer.x)

                # Check if the user is looking at the screen (gaze ratio is near center)
                is_facing_screen = 0.3 < gaze_ratio < 0.7  # Increased tolerance

                # Shoulder alignment check
                shoulder_diff = abs(left_shoulder[1] - right_shoulder[1])
                is_upright = shoulder_diff < 0.08  # Increased tolerance

                # Debug print statements
                # print(f"Gaze Ratio: {gaze_ratio:.2f}, Shoulder Diff: {shoulder_diff:.2f}, Nose Y: {nose[1]:.2f}")

                if is_facing_screen and is_upright and 0.3 < nose[1] < 0.7:
                    stage = "Active Listening"

        except Exception as e:
            print(f"Error processing frame: {e}")

        # Display the stage
        cv2.rectangle(image, (480, 20), (630, 120), (19, 69, 139), -1)
        cv2.putText(image, 'STAGE', (490, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, stage, (490, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.imshow('Mediapipe Feed', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
