"""
Face Recognition App - Streamlit version
Converted from the original Tkinter desktop app.

Run with:
    streamlit run face_app_streamlit.py

Notes on the conversion:
- Tkinter's blocking `while True` webcam loops (with cv2.imshow) are replaced
  with a Streamlit-friendly loop that writes frames to an `st.image`
  placeholder and checks a "Stop" button each iteration.
- All persistent state (name/age/gender fields, capture progress, running
  flags) lives in `st.session_state` since Streamlit reruns the whole script
  on every interaction.
- File layout is unchanged: face crops go in `data/`, identities are tracked
  in `user_names.csv`, and the trained model is written to `classifier.xml`.
"""

import csv
import glob
import os

import cv2
import numpy as np
import streamlit as st
from PIL import Image

DATA_DIR = "data"
USER_CSV = "user_names.csv"
CLASSIFIER_FILE = "classifier.xml"

os.makedirs(DATA_DIR, exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if face_cascade.empty():
    fallback_path = os.path.join(os.getcwd(), "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(fallback_path)
if face_cascade.empty():
    st.error("Unable to load Haar cascade classifier. Place haarcascade_frontalface_default.xml next to this script.")
    st.stop()


# ---------------------------------------------------------------------------
# Shared helpers (ported ~1:1 from the Tkinter app)
# ---------------------------------------------------------------------------

def detect_faces(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.equalizeHist(gray_img)
    faces = face_cascade.detectMultiScale(
        gray_img,
        scaleFactor=1.08,
        minNeighbors=8,
        minSize=(70, 70),
    )
    if len(faces) == 0:
        return []
    return sorted(faces, key=lambda rect: rect[2] * rect[3], reverse=True)


def load_user_names(csv_file=USER_CSV):
    rows = []
    if os.path.exists(csv_file):
        with open(csv_file, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
    return [row for row in rows if row]


def save_user_names(rows, csv_file=USER_CSV):
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def face_cropped(img):
    faces = detect_faces(img)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    face = img[y:y + h, x:x + w]
    if face.size == 0:
        return None
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.equalizeHist(face)
    face = cv2.resize(face, (200, 200))
    return cv2.GaussianBlur(face, (3, 3), 0)


def next_face_id(user_rows):
    if len(user_rows) == 0:
        return 1
    return int(user_rows[-1][0]) + 1


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

for key, default in {
    "capturing": False,
    "detecting": False,
    "img_id": 0,
    "current_face_id": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


st.set_page_config(page_title="Face Recognition App", page_icon="🧑‍💻", layout="centered")
st.title("🧑‍💻 Face Recognition App")

tab_register, tab_train, tab_detect, tab_manage = st.tabs(
    ["➕ Register & Capture", "🎯 Train", "🔍 Detect", "🗑️ Manage Users"]
)

# ---------------------------------------------------------------------------
# Tab 1: Register + Capture dataset
# ---------------------------------------------------------------------------
with tab_register:
    st.subheader("Register a new user")

    col1, col2, col3 = st.columns(3)
    with col1:
        name = st.text_input("Name")
    with col2:
        age = st.text_input("Age")
    with col3:
        gender = st.selectbox("Gender", ["Male", "Female"])

    num_samples = st.slider("Number of face samples to capture", 50, 600, 200, step=50)

    start_col, stop_col = st.columns(2)
    start_capture = start_col.button("Start Capture", type="primary", use_container_width=True)
    stop_capture = stop_col.button("Stop Capture", use_container_width=True)

    frame_placeholder = st.empty()
    progress_placeholder = st.empty()

    if start_capture:
        if not name or not age:
            st.error("Please fill in Name and Age before capturing.")
        else:
            user_rows = load_user_names()
            face_id = next_face_id(user_rows)
            user_rows.append([str(face_id), name, age, gender])
            save_user_names(user_rows)

            st.session_state.current_face_id = face_id
            st.session_state.img_id = 0
            st.session_state.capturing = True

    if stop_capture:
        st.session_state.capturing = False

    if st.session_state.capturing:
        face_id = st.session_state.current_face_id
        cap = cv2.VideoCapture(0)
        try:
            while st.session_state.capturing and st.session_state.img_id < num_samples:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Could not read from webcam.")
                    break

                cropped = face_cropped(frame)
                display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if cropped is not None:
                    st.session_state.img_id += 1
                    file_name_path = os.path.join(
                        DATA_DIR, f"user.{face_id}.{st.session_state.img_id}.jpg"
                    )
                    cv2.imwrite(file_name_path, cropped)

                frame_placeholder.image(display_frame, channels="RGB")
                progress_placeholder.progress(
                    min(st.session_state.img_id / num_samples, 1.0),
                    text=f"Captured {st.session_state.img_id}/{num_samples} samples",
                )
        finally:
            cap.release()

        st.session_state.capturing = False
        if st.session_state.img_id > 0:
            st.success(f"Collecting samples completed! Saved {st.session_state.img_id} images for user {face_id}.")
        st.rerun()

# ---------------------------------------------------------------------------
# Tab 2: Train classifier
# ---------------------------------------------------------------------------
with tab_train:
    st.subheader("Train the recognizer")
    st.write("Trains an LBPH face recognizer on every image currently in the `data/` folder.")

    if st.button("Train Classifier", type="primary"):
        if not os.listdir(DATA_DIR):
            st.error("No training data found in the data folder.")
        else:
            with st.spinner("Training..."):
                paths = [os.path.join(DATA_DIR, file) for file in os.listdir(DATA_DIR)]
                faces = []
                ids = []

                for image_path in paths:
                    if not os.path.isfile(image_path):
                        continue
                    img = Image.open(image_path).convert("L")
                    image_np = np.array(img, "uint8")
                    face_id = int(os.path.split(image_path)[1].split(".")[1])
                    faces.append(image_np)
                    ids.append(face_id)

                ids = np.array(ids)

                clf = cv2.face.LBPHFaceRecognizer_create()
                clf.train(faces, ids)
                clf.write(CLASSIFIER_FILE)

            st.success("Training completed!")

# ---------------------------------------------------------------------------
# Tab 3: Live detection
# ---------------------------------------------------------------------------
with tab_detect:
    st.subheader("Live recognition")

    if not os.path.exists(CLASSIFIER_FILE):
        st.info("No trained classifier found yet. Train one in the 'Train' tab first.")
    else:
        start_col, stop_col = st.columns(2)
        start_detect = start_col.button("Start Detection", type="primary", use_container_width=True)
        stop_detect = stop_col.button("Stop Detection", use_container_width=True)

        detect_placeholder = st.empty()

        if start_detect:
            st.session_state.detecting = True
        if stop_detect:
            st.session_state.detecting = False

        if st.session_state.detecting:
            clf = cv2.face.LBPHFaceRecognizer_create()
            clf.read(CLASSIFIER_FILE)
            user_rows = load_user_names()
            id_to_name = {row[0]: (row[1] if len(row) > 1 else "Unknown") for row in user_rows}

            cap = cv2.VideoCapture(0)
            try:
                while st.session_state.detecting:
                    ret, img = cap.read()
                    if not ret:
                        st.warning("Could not read from webcam.")
                        break

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
                        face_id, pred = clf.predict(face_roi)
                        confidence = int(100 * (1 - pred / 300))

                        if confidence > 70:
                            label = f"{id_to_name.get(str(face_id), 'Unknown')} ({confidence}%)"
                            color = (255, 255, 255)
                        else:
                            label = f"UNKNOWN ({confidence}%)"
                            color = (0, 0, 255)

                        cv2.putText(img, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

                    display_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    detect_placeholder.image(display_frame, channels="RGB")
            finally:
                cap.release()

# ---------------------------------------------------------------------------
# Tab 4: Manage / delete users
# ---------------------------------------------------------------------------
with tab_manage:
    st.subheader("Registered users")
    user_rows = load_user_names()
    if user_rows:
        st.table(
            [
                {
                    "ID": row[0],
                    "Name": row[1] if len(row) > 1 else "",
                    "Age": row[2] if len(row) > 2 else "",
                    "Gender": row[3] if len(row) > 3 else "",
                }
                for row in user_rows
            ]
        )
    else:
        st.write("No users registered yet.")

    st.subheader("Delete a user")
    identifier = st.text_input("User ID or Name to delete")

    if st.button("Delete", type="secondary"):
        identifier = identifier.strip()
        if identifier == "":
            st.error("Please enter a user ID or name.")
        else:
            rows = load_user_names()
            target_row = None
            target_id = None

            if identifier.isdigit():
                for row in rows:
                    if row and row[0] == identifier:
                        target_row = row
                        target_id = identifier
                        break
            else:
                for row in rows:
                    if row and len(row) > 1 and row[1].strip().lower() == identifier.lower():
                        target_row = row
                        target_id = row[0]
                        break

            if target_row is None:
                st.error("User not found.")
            else:
                updated_rows = [row for row in rows if row != target_row]
                save_user_names(updated_rows)

                deleted_files = []
                for file_path in glob.glob(os.path.join(DATA_DIR, f"user.{target_id}.*")):
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        deleted_files.append(os.path.basename(file_path))

                st.success(f"Deleted user {target_id} and {len(deleted_files)} file(s).")
                st.rerun()

# BY - Sayan Roy (converted to Streamlit)
