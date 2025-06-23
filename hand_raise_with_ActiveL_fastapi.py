import base64
from io import BytesIO
from fastapi import FastAPI, WebSocket, WebSocketDisconnect # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import cv2
import numpy as np
import mediapipe as mp # type: ignore
import logging
import socketio # type: ignore
from fastapi import FastAPI # type: ignore
from fastapi_socketio import SocketManager # type: ignore
import uvicorn

# Suppress Mediapipe logs
logging.getLogger("mediapipe").setLevel(logging.ERROR)

# Initialize FastAPI app
app = FastAPI()

# Create a Socket.IO AsyncServer instance
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins='*',
    logger=True,
    engineio_logger=True,
    ping_timeout=60,
    ping_interval=25,
    transports=['polling', 'websocket']  # Enable both polling and websocket
)

# Dictionary to hold connected users
connected_users = {}


# Create ASGIApp
socket_app = socketio.ASGIApp(
    socketio_server=sio,
    other_asgi_app=app,
    socketio_path='socket.io',
    static_files={
        '/': './static'  # Optional: serve static files if needed
    }
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Mediapipe models
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

# Active connections dictionary
active_connections = {}

def decode_frame(encoded_frame):
    """Decode base64-encoded image frame."""
    try:
        if encoded_frame.startswith("data:image"):
            encoded_frame = encoded_frame.split(",")[1]
        data = base64.b64decode(encoded_frame)
        img = Image.open(BytesIO(data))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        # print(f"Error decoding frame: {e}")
        return None

def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180 else 360 - angle

stage = "Unknown"

def process_frame(frame):
    """Process frame for pose and gaze detection."""
    global stage
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:


            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            pose_results = pose.process(image)
            face_results = face_mesh.process(image)
            
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            
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
                return "Student left"

            return stage

# Add a health check endpoint
@app.get("/")
async def health_check():
    return {
        "status": "Server is running", 
        "socket_io_path": "/socket.io",
        "connection_example": """
        // Replace YOUR_SERVER_IP with your actual server IP address
        const socket = io('https://YOUR_SERVER_IP:8443', {
            path: '/socket.io',
            transports: ['polling', 'websocket'],
            secure: true,
            rejectUnauthorized: false
        });
        """
    }

@app.get("/api/info")
async def get_info():
    return {
        "server_url": "https://YOUR_SERVER_IP:8443",  # Replace YOUR_SERVER_IP with your actual server IP
        "socket_io": {
            "path": "/socket.io",
            "connection": {
                "url": "https://YOUR_SERVER_IP:8443",  # Replace YOUR_SERVER_IP with your actual server IP
                "options": {
                    "path": "/socket.io",
                    "transports": ["polling", "websocket"],
                    "secure": True,
                    "rejectUnauthorized": False
                }
            },
            "events": {
                "incoming": {
                    "frame": "Send base64 encoded frame for processing"
                },
                "outgoing": {
                    "status": "Receives detection status updates",
                    "connect": "Emitted when connected",
                    "disconnect": "Emitted when disconnected"
                }
            }
        }
    }

@sio.event
async def connect(sid, environ):
    print(f"Client {sid} connected")
    await sio.emit('status', {'status': 'Connected'}, room=sid)
    print(f"Sent connection confirmation to {sid}")

@sio.event
async def disconnect(sid):
    print(f"Client {sid} disconnected")

@sio.event
async def register(sid, data):
    username = data.get('username')
    print(f"User {username} registered with sid {sid}")
    active_connections[sid] = username
    connected_users[sid] = username

    await sio.emit('status', {'status': 'Registered'}, room=sid)

@sio.event
async def send_frame(sid, data):
    try:
        print(f"Received frame from client {sid}")
        print(f"{connected_users}")
        sender = data.get('sender')
        receiver = data.get('receiver')
        frame_data = data.get('frame')

        receiver_socket_id = None
        for sid, username in connected_users.items():
            if username == receiver:
                receiver_socket_id = sid
                break
        # await sio.emit('response',"user joined"+sid,receiver_socket_id)

        if not frame_data:
            print(f"No frame data received from {sid}")
            return

        frame = decode_frame(frame_data)
        if frame is None:
            print(f"Error decoding frame from {sid}")
            return

        result = process_frame(frame)
        print(f"Processed frame for {sid}, result: {result}")

        # Send response in the format expected by client
        # result={"status":"success"}
        response_data = {
            "sender": sender,
            "receiver": receiver,
            "data": result
        }
        await sio.emit('response', response_data, room=receiver_socket_id)
        print(f"Sent response to {sid}: {result}")

    except Exception as e:
        error_msg = f"Error processing frame: {str(e)}"
        print(f"Error for {sid}: {error_msg}")
        await sio.emit('error', {"error": error_msg}, room=receiver_socket_id)

if __name__=="__main__":
    print("Starting server...")  # Debug log
    uvicorn.run(
        socket_app,
        host="0.0.0.0",
        port=8443,
        workers=1,
        ssl_keyfile="C:/Users/VR/Desktop/Hybrid_learning/AI/project_env/keys/onsite.key",
        ssl_certfile='C:/Users/VR/Desktop/Hybrid_learning/AI/project_env/keys/onsite.crt',
        log_level="debug"
    )