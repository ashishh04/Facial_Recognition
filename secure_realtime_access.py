import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist
from sklearn.metrics.pairwise import cosine_similarity
import insightface
import mysql.connector

# === CONFIG ===
SIMILARITY_THRESHOLD = 0.55
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 3

# === MySQL Configuration ===
MYSQL_CONFIG = {
    'host': 'localhost',
    'user': 'root',       # ✅ Replace with your MySQL username
    'password': '12345',  # ✅ Replace with your MySQL password
    'database': 'facial_access_db'   # ✅ Replace with your MySQL database
}

# === Logging Function ===
def log_access(person_name, method, success, reason):
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cur = conn.cursor()
        query = '''
            INSERT INTO access_logs (person_name, method, success, reason)
            VALUES (%s, %s, %s, %s)
        '''
        cur.execute(query, (person_name, method, success, reason))
        conn.commit()
        cur.close()
        conn.close()
        print(f"[LOGGED] {person_name} - {'GRANTED' if success else 'DENIED'} - {reason}")
    except mysql.connector.Error as err:
        print(f"[ERROR] MySQL: {err}")

# === ArcFace Setup ===
model = insightface.app.FaceAnalysis(name='buffalo_l')
model.prepare(ctx_id=0)
known_faces = np.load("arcface_embeddings.npy", allow_pickle=True).item()

# === Dlib Setup for Blink Detection ===
predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# === EAR Function ===
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# === Main Loop ===
cap = cv2.VideoCapture(0)
blink_counter = 0
blink_total = 0
last_detected_name = "Unknown"

print("[INFO] Starting secure recognition system...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face Detection (ArcFace)
    faces_arc = model.get(rgb)

    # Blink Detection (Dlib)
    faces_dlib = detector(gray)
    liveness_passed = False

    for face_d in faces_dlib:
        shape = predictor(gray, face_d)
        shape = face_utils.shape_to_np(shape)
        left_eye = shape[lStart:lEnd]
        right_eye = shape[rStart:rEnd]

        left_EAR = eye_aspect_ratio(left_eye)
        right_EAR = eye_aspect_ratio(right_eye)
        ear = (left_EAR + right_EAR) / 2.0

        if ear < EAR_THRESHOLD:
            blink_counter += 1
        else:
            if blink_counter >= CONSEC_FRAMES:
                blink_total += 1
                liveness_passed = True
            blink_counter = 0

    # Face Recognition
    if faces_arc:
        face = faces_arc[0]
        emb = face.embedding
        bbox = face.bbox.astype(int)
        name = "Unknown"

        for person_name, known_emb in known_faces.items():
            similarity = cosine_similarity([emb], [known_emb])[0][0]
            if similarity > SIMILARITY_THRESHOLD:
                name = person_name
                break

        last_detected_name = name
        label = f"{name} - {'LIVE' if liveness_passed else 'SPOOF?'}"
        color = (0, 255, 0) if liveness_passed else (0, 0, 255)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # === Log Access to MySQL ===
        if name != "Unknown" and liveness_passed:
            log_access(name, "face+liveness", True, "Access granted")
        else:
            log_access(name, "face+liveness", False, "No match or spoof detected")

    # Show blink count
    cv2.putText(frame, f"Blinks: {blink_total}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Secure Face Access", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
