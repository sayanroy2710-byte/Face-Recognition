"""
Face Recognition — Streamlit + streamlit-webrtc

Continuous webcam feed in the browser (via WebRTC), replacing the original
Tkinter + cv2.VideoCapture(0) + cv2.imshow desktop loop.

IMPORTANT ARCHITECTURE NOTES (read before deploying):

1. MODEL FORMAT: cv2.face.LBPHFaceRecognizer is a C++ object — it cannot be
   pickled with Python's `pickle` module (no __reduce__/__getstate__).
   It has its own serialization: clf.write("classifier.xml") / clf.read(...).
   This app sticks to classifier.xml for that reason. If you truly need a
   single .pkl artifact (e.g. to fit an existing pipeline), see the
   `pack_model_as_pkl` / `unpack_model_from_pkl` helpers at the bottom —
   they just pickle the raw XML bytes, there's no other way to do it.

2. DEPENDENCY: cv2.face requires `opencv-contrib-python`, NOT plain
   `opencv-python`. Installing the wrong one is the #1 reason this kind of
   app fails after deployment with `AttributeError: module 'cv2' has no
   attribute 'face'`.

3. PERSISTENCE: Streamlit Community Cloud's filesystem is EPHEMERAL — every
   redeploy / restart wipes anything written to disk (data/, classifier.xml,
   user_names.csv). For a real deployment, point the save/load paths below
   at persistent storage (S3, a mounted volume, a database) instead of the
   local `data/` folder. Locally, or on a host with a real persistent disk,
   the local-folder version below works as-is.

4. WEBRTC ON CLOUD: streamlit-webrtc needs a STUN server (and, on some
   networks, a TURN server) to establish the peer connection once it's not
   running on localhost. This app sets a public Google STUN server in
   RTC_CONFIGURATION — for restrictive networks you may need to add a TURN
   server (e.g. Twilio's free/paid TURN) for the video to connect at all.
"""

import csv
import glob
import os
import pickle
import threading

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

DATA_DIR = "data"
CSV_FILE = "user_names.csv"
MODEL_FILE = "classifier.xml"

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

os.makedirs(DATA_DIR, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ---------------------------------------------------------------------------
# Shared helpers (ported from the notebook, unchanged in logic)
# ---------------------------------------------------------------------------

def detect_faces(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.equalizeHist(gray_img)
    faces = face_cascade.detectMultiScale(
        gray_img, scaleFactor=1.08, minNeighbors=8, minSize=(70, 70)
    )
    if len(faces) == 0:
        return []
    return sorted(faces, key=lambda rect: rect[2] * rect[3], reverse=True)


def load_user_names(csv_file=CSV_FILE):
    rows = []
    if os.path.exists(csv_file):
        with open(csv_file, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
    return [row for row in rows if row]


def save_user_names(rows, csv_file=CSV_FILE):
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def next_face_id():
    rows = load_user_names()
    if not rows:
        return 1
    return int(rows[-1][0]) + 1


def delete_user_data(identifier: str) -> str:
    identifier = identifier.strip()
    if not identifier:
        return "Please enter a user ID or name."

    user_rows = load_user_names()
    target_row, target_id = None, None

    if identifier.isdigit():
        for row in user_rows:
            if row and row[0] == identifier:
                target_row, target_id = row, identifier
                break
    else:
        for row in user_rows:
            if row and len(row) > 1 and row[1].strip().lower() == identifier.lower():
                target_row, target_id = row, row[0]
                break

    if target_row is None:
        return "User not found."

    save_user_names([row for row in user_rows if row != target_row])

    deleted = 0
    for file_path in glob.glob(os.path.join(DATA_DIR, f"user.{target_id}.*")):
        if os.path.isfile(file_path):
            os.remove(file_path)
            deleted += 1

    return f"Deleted user {target_id} and {deleted} file(s)."


def train_classifier() -> str:
    if not os.listdir(DATA_DIR):
        return "No training data found in the data folder."

    faces, ids = [], []
    for image_path in [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)]:
        if not os.path.isfile(image_path):
            continue
        img = Image.open(image_path).convert("L")
        image_np = np.array(img, "uint8")
        face_id = int(os.path.split(image_path)[1].split(".")[1])
        faces.append(image_np)
        ids.append(face_id)

    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, np.array(ids))
    clf.write(MODEL_FILE)
    return "Training completed."


# Optional: pack/unpack the trained model as a single .pkl if some other
# part of your pipeline genuinely requires that extension. This does NOT
# pickle the recognizer object — it pickles the XML bytes it writes itself.
def pack_model_as_pkl(xml_path=MODEL_FILE, pkl_path="classifier.pkl"):
    with open(xml_path, "rb") as f:
        xml_bytes = f.read()
    with open(pkl_path, "wb") as f:
        pickle.dump({"format": "lbph-xml", "data": xml_bytes}, f)


def unpack_model_from_pkl(pkl_path="classifier.pkl", xml_path=MODEL_FILE):
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    with open(xml_path, "wb") as f:
        f.write(payload["data"])


# ---------------------------------------------------------------------------
# WebRTC video processors
# ---------------------------------------------------------------------------

class CaptureProcessor:
    """Crops+saves faces to data/ while a webrtc stream is running.

    streamlit-webrtc runs recv() on a background thread, so state is kept
    on the processor instance itself (not st.session_state) and read back
    from the main thread via ctx.video_processor after the stream starts.
    """

    def __init__(self, face_id: int, target_count: int):
        self.face_id = face_id
        self.target_count = target_count
        self.img_id = 0
        self.lock = threading.Lock()
        self.done = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        with self.lock:
            if not self.done:
                faces = detect_faces(img)
                if faces:
                    x, y, w, h = faces[0]
                    face = img[y:y + h, x:x + w]
                    if face.size:
                        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                        gray = cv2.equalizeHist(gray)
                        gray = cv2.resize(gray, (200, 200))
                        gray = cv2.GaussianBlur(gray, (3, 3), 0)
                        self.img_id += 1
                        cv2.imwrite(
                            os.path.join(DATA_DIR, f"user.{self.face_id}.{self.img_id}.jpg"),
                            gray,
                        )
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(
                            img, f"{self.img_id}/{self.target_count}", (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                        )
                    if self.img_id >= self.target_count:
                        self.done = True

        return av.VideoFrame.from_ndarray(img, format="bgr24")


class RecognizeProcessor:
    def __init__(self):
        self.clf = cv2.face.LBPHFaceRecognizer_create()
        self.clf.read(MODEL_FILE)
        self.user_rows = load_user_names()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.equalizeHist(gray_img)

        faces = detect_faces(img)
        for (x, y, w, h) in faces[:1]:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_roi = gray_img[y:y + h, x:x + w]
            if face_roi.size == 0:
                continue
            face_roi = cv2.resize(face_roi, (200, 200))
            face_roi = cv2.GaussianBlur(face_roi, (3, 3), 0)
            face_id, pred = self.clf.predict(face_roi)
            confidence = int(100 * (1 - pred / 300))

            if confidence > 70:
                name = "Unknown"
                for row in self.user_rows:
                    if row and row[0] == str(face_id):
                        name = row[1] if len(row) > 1 else "Unknown"
                        break
                label, color = f"{name} ({confidence}%)", (255, 255, 255)
            else:
                label, color = f"UNKNOWN ({confidence}%)", (0, 0, 255)

            cv2.putText(img, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Face Recognition", page_icon="🧑‍💻")
st.title("Face Recognition")

mode = st.sidebar.radio("Mode", ["Register (Capture)", "Train", "Recognize", "Delete User"])

if mode == "Register (Capture)":
    st.subheader("Register a new user")
    name = st.text_input("Name")
    age = st.text_input("Age")
    gender = st.selectbox("Gender", ["Male", "Female"])
    target_count = st.slider("Number of samples to capture", 20, 300, 100, step=10)

    if "capture_face_id" not in st.session_state:
        st.session_state.capture_face_id = None

    start = st.button("Start capture", disabled=not (name and age))
    if start:
        face_id = next_face_id()
        rows = load_user_names()
        rows.append([str(face_id), name, age, gender])
        save_user_names(rows)
        st.session_state.capture_face_id = face_id
        st.success(f"Registered as user #{face_id}. Start the stream below and look at the camera.")

    if st.session_state.capture_face_id is not None:
        # Read session_state into a plain local BEFORE building the factory.
        # video_processor_factory is invoked by streamlit-webrtc on a
        # background worker thread that has no ScriptRunContext, so
        # referencing st.session_state *inside* the closure raises
        # AttributeError. Closing over a local variable avoids that.
        face_id_for_capture = st.session_state.capture_face_id

        ctx = webrtc_streamer(
            key="capture",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=lambda: CaptureProcessor(
                face_id_for_capture, target_count
            ),
            media_stream_constraints={"video": True, "audio": False},
        )
        if ctx.video_processor:
            st.info(f"Captured: {ctx.video_processor.img_id} / {target_count}")
            if ctx.video_processor.done:
                st.success("Target sample count reached — you can stop the stream now.")

elif mode == "Train":
    st.subheader("Train classifier")
    if st.button("Train now"):
        st.info(train_classifier())

elif mode == "Recognize":
    st.subheader("Live recognition")
    if not os.path.exists(MODEL_FILE):
        st.warning("No trained model found yet — register users and train first.")
    else:
        webrtc_streamer(
            key="recognize",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=RecognizeProcessor,
            media_stream_constraints={"video": True, "audio": False},
        )

elif mode == "Delete User":
    st.subheader("Delete a user")
    identifier = st.text_input("User ID or Name")
    if st.button("Delete"):
        st.info(delete_user_data(identifier))
