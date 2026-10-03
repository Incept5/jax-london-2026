import sys
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Put one photo per person in day-1/faces/, named after them (e.g. faces/John.jpg).
# The folder is git-ignored: only use photos of people who have agreed.
FACES = Path(__file__).parent / "faces"
MATCH = 0.35   # cosine similarity above this = same person (ArcFace: same person ~0.5-0.8, others ~0.1-0.3)

app = FaceAnalysis(providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])   # ~280 MB model, first run
app.prepare(ctx_id=0, det_size=(640, 640))


def biggest_face(img):
    faces = app.get(img)
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])) if faces else None


# 1. Enrol: one 512-number embedding per reference photo
known = {}
for p in sorted(FACES.glob("*.*")):
    img = cv2.imread(str(p))
    face = biggest_face(img) if img is not None else None
    if face is not None:
        known[p.stem] = face.normed_embedding
        print("enrolled", p.stem)
if not known:
    sys.exit(f"Put a photo of yourself in {FACES} first (e.g. faces/YourName.jpg)")


def label(frame):
    """Draw a box and the best-matching name (cosine similarity, as in embedding_demo.py) on every face."""
    for face in app.get(frame):
        name, score = max(((n, float(np.dot(face.normed_embedding, e))) for n, e in known.items()), key=lambda x: x[1])
        x1, y1, x2, y2 = map(int, face.bbox)
        colour = (0, 200, 0) if score > MATCH else (0, 165, 255)
        text = f"{name} {score:.2f}" if score > MATCH else f"unknown ({name} {score:.2f})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(frame, text, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
        print(text)
    return frame


# 2. Match: python face_recognition_demo.py photo.jpg  (a photo)  or no argument (the webcam, q to quit)
if len(sys.argv) > 1:
    cv2.imwrite("labelled.jpg", label(cv2.imread(sys.argv[1])))
    print("wrote labelled.jpg")
else:
    cam = cv2.VideoCapture(0)       # try 1 if an iPhone grabs camera 0 (Continuity Camera)
    while True:
        ok, frame = cam.read()
        if not ok:
            sys.exit("No camera frames: allow camera access for your terminal in System Settings > Privacy")
        cv2.imshow("Face recognition (q to quit)", label(frame))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
