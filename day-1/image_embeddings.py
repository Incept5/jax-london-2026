import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).parent
PHOTOS = sorted((HERE.parent / "day-2" / "data").glob("*.jp*g"))

# CLIP puts images AND text into the same embedding space, so a sentence can find a photo.
model = SentenceTransformer("clip-ViT-B-32")

QUERIES = [
    "people drinking beer",
    "a restaurant menu",
    "a bar chart",
    "a building or architecture drawing",
]


def photo_search():
    images = model.encode([Image.open(p).convert("RGB") for p in PHOTOS], normalize_embeddings=True)
    for query in (sys.argv[1:] or QUERIES):
        q = model.encode(query, normalize_embeddings=True)
        scores = images @ q                      # cosine similarity, as in embedding_demo.py
        best = np.argsort(-scores)[:3]
        print(f"\n{query!r}")
        for i in best:
            print(f"   {scores[i]:.3f}  {PHOTOS[i].name}")


def face_match():
    # Faces of public figures from the LFW dataset (downloads ~200 MB the first time)
    from sklearn.datasets import fetch_lfw_people
    lfw = fetch_lfw_people(min_faces_per_person=40, color=True, resize=1.0)
    faces = [Image.fromarray((img * 255).astype("uint8") if img.max() <= 1 else img.astype("uint8")) for img in lfw.images]
    vecs = model.encode(faces, normalize_embeddings=True, batch_size=64)
    rng = np.random.default_rng(1)
    hits = 0
    for probe in rng.choice(len(faces), 10, replace=False):
        scores = vecs @ vecs[probe]
        scores[probe] = -1                      # don't match the photo with itself
        match = int(np.argmax(scores))
        same = lfw.target[match] == lfw.target[probe]
        hits += same
        print(f"{lfw.target_names[lfw.target[probe]]:25} -> {lfw.target_names[lfw.target[match]]:25} "
              f"{scores[match]:.3f} {'same person' if same else 'different person'}")
    print(f"\n{hits}/10 nearest faces were the same person")


if __name__ == "__main__":
    if sys.argv[1:] == ["--faces"]:
        face_match()
    else:
        photo_search()
