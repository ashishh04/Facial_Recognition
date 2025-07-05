import cv2
import face_recognition
import numpy as np

CAPTURE_FRAMES = 5
MOVEMENT_THRESHOLD = 5  # pixels

# Initialize video
video = cv2.VideoCapture(0)
print("[INFO] Move your head slightly to prove liveness.")

frames = []
while len(frames) < CAPTURE_FRAMES:
    ret, frame = video.read()
    if not ret: break
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(rgb)
    if faces:
        top, right, bottom, left = faces[0]
        cx = (left + right) // 2
        cy = (top + bottom) // 2
        frames.append((cx, cy))
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
    
    cv2.imshow("Move your head", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        video.release()
        cv2.destroyAllWindows()
        exit(0)

video.release()
cv2.destroyAllWindows()

if len(frames) < CAPTURE_FRAMES:
    print("❌ Liveness failed: not enough frames")
else:
    deltas = [np.linalg.norm(np.subtract(frames[i], frames[i-1])) for i in range(1, len(frames))]
    print("Motion detected:", deltas)
    if max(deltas) > MOVEMENT_THRESHOLD:
        print("✅ Liveness confirmed (face movement detected)")
    else:
        print("❌ Liveness failed: no significant movement")
import cv2
import face_recognition
import numpy as np

CAPTURE_FRAMES = 5
MOVEMENT_THRESHOLD = 5  # pixels

# Initialize video
video = cv2.VideoCapture(0)
print("[INFO] Move your head slightly to prove liveness.")

frames = []
while len(frames) < CAPTURE_FRAMES:
    ret, frame = video.read()
    if not ret: break
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(rgb)
    if faces:
        top, right, bottom, left = faces[0]
        cx = (left + right) // 2
        cy = (top + bottom) // 2
        frames.append((cx, cy))
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
    
    cv2.imshow("Move your head", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        video.release()
        cv2.destroyAllWindows()
        exit(0)

video.release()
cv2.destroyAllWindows()

if len(frames) < CAPTURE_FRAMES:
    print("❌ Liveness failed: not enough frames")
else:
    deltas = [np.linalg.norm(np.subtract(frames[i], frames[i-1])) for i in range(1, len(frames))]
    print("Motion detected:", deltas)
    if max(deltas) > MOVEMENT_THRESHOLD:
        print("✅ Liveness confirmed (face movement detected)")
    else:
        print("❌ Liveness failed: no significant movement")
