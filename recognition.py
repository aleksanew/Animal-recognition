from pathlib import Path
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os, sys, contextlib, time
from collections import deque
from datetime import datetime

from sympy import false

# ------------------ config ------------------
BASE = Path("PetImages")
CLASSES = ["Cat", "Dog"]                 # 0=Cat, 1=Dog
IMG_SIZE = (160, 160)                    # bigger field for shape cues
BUNDLE_PATH = "catdog_hog_svm.joblib"

# HOG params (coarser; reduces background/texture bias)
HOG_ORIENT = 12
HOG_PIX_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (3, 3)
HOG_BLOCK_STRIDE = (16, 16)              # multiple of cell size
USE_CLAHE = True                         # lighting robust
# --------------------------------------------

def make_hog(img_size=IMG_SIZE,
             orientations=HOG_ORIENT,
             pixels_per_cell=HOG_PIX_PER_CELL,
             cells_per_block=HOG_CELLS_PER_BLOCK,
             block_stride=HOG_BLOCK_STRIDE):
    # OpenCV HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    win = (img_size[1], img_size[0])  # (w, h)
    block = (cells_per_block[0]*pixels_per_cell[0],
             cells_per_block[1]*pixels_per_cell[1])
    stride = block_stride
    cell = pixels_per_cell
    bins = orientations
    return cv2.HOGDescriptor(win, block, stride, cell, bins)

@contextlib.contextmanager
def suppress_stderr():
    saved = sys.stderr
    try:
        sys.stderr = open(os.devnull, 'w')
        yield
    finally:
        try: sys.stderr.close()
        except Exception: pass
        sys.stderr = saved

def safe_imread(path):
    with suppress_stderr():
        return cv2.imread(path)

def iter_image_paths(base=BASE, classes=CLASSES):
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    X, y = [], []
    for label, cls in enumerate(classes):
        for p in (base / cls).rglob("*"):
            if p.suffix in exts:
                X.append(str(p))
                y.append(label)
    return np.array(X), np.array(y, dtype=int)

# ---------- preprocessing & features ----------
def preprocess_gray(img, size=IMG_SIZE, use_clahe=USE_CLAHE):
    if img is None: raise ValueError("preprocess_gray: img is None")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
    return gray

def featurize_gray(gray, hog_desc):
    return hog_desc.compute(gray).ravel()

def featurize_path(path, hog_desc, size=IMG_SIZE, use_clahe=USE_CLAHE):
    img = safe_imread(path)
    if img is None:
        raise ValueError(f"Could not read image {path}")
    gray = preprocess_gray(img, size=size, use_clahe=use_clahe)
    return featurize_gray(gray, hog_desc)

def build_matrix(paths, hog_desc, size=IMG_SIZE, use_clahe=USE_CLAHE):
    feats, kept = [], []
    for i, p in enumerate(paths):
        try:
            feats.append(featurize_path(p, hog_desc, size=size, use_clahe=use_clahe))
            kept.append(i)
        except Exception:
            pass
    return (np.vstack(feats), np.array(kept, dtype=int))

# ---------- training ----------
def train_and_save(bundle=BUNDLE_PATH):
    X_paths, y = iter_image_paths()
    print(f"Found {len(X_paths)} files - Cats: {np.sum(y==0)}  Dogs: {np.sum(y==1)}")

    Xtrainp, Xtestp, ytrain, ytest = train_test_split(
        X_paths, y, test_size=0.2, random_state=42, stratify=y
    )

    hog = make_hog(IMG_SIZE, HOG_ORIENT, HOG_PIX_PER_CELL, HOG_CELLS_PER_BLOCK, HOG_BLOCK_STRIDE)

    Xtrain, keep_tr = build_matrix(Xtrainp, hog, size=IMG_SIZE, use_clahe=USE_CLAHE)
    Xtest, keep_te = build_matrix(Xtestp, hog, size=IMG_SIZE, use_clahe=USE_CLAHE)
    ytrain = ytrain[keep_tr]
    ytest  = ytest[keep_te]

    dual_mode = not (Xtrain.shape[0] > Xtrain.shape[1])  # True if n_samples < n_features

    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("svm", LinearSVC(
            C=1.0, class_weight="balanced", loss="squared_hinge",
            dual=dual_mode, tol=1e-3, max_iter=10000, random_state=42
        ))
    ])

    print("Starting SVM training...")
    start = time.time()
    clf.fit(Xtrain, ytrain)
    print(f"Training took {time.time()-start:.2f}s")
    ypred = clf.predict(Xtest)
    print(confusion_matrix(ytest, ypred))
    print(classification_report(ytest, ypred, target_names=CLASSES))

    joblib.dump({
        "model": clf,
        "classes": CLASSES,
        "img_size": IMG_SIZE,
        "hog_params": {
            "orientations": HOG_ORIENT,
            "pixels_per_cell": HOG_PIX_PER_CELL,
            "cells_per_block": HOG_CELLS_PER_BLOCK,
            "block_stride": HOG_BLOCK_STRIDE,
        },
        "use_clahe": USE_CLAHE
    }, bundle)
    print(f"Saved model to {bundle}")

# ---------- inference ----------
def _load_bundle(bundle=BUNDLE_PATH):
    b = joblib.load(bundle)
    size = tuple(b.get("img_size", IMG_SIZE))
    hp = b.get("hog_params", {})
    hog = make_hog(size,
                   hp.get("orientations", HOG_ORIENT),
                   hp.get("pixels_per_cell", HOG_PIX_PER_CELL),
                   hp.get("cells_per_block", HOG_CELLS_PER_BLOCK),
                   hp.get("block_stride", HOG_BLOCK_STRIDE))
    return b, hog

def predict_image(path, bundle=BUNDLE_PATH, verbose=True):
    b, hog = _load_bundle(bundle)
    model, classes = b["model"], b["classes"]
    size, use_clahe = tuple(b["img_size"]), b.get("use_clahe", USE_CLAHE)

    img = safe_imread(path)
    if img is None:
        raise ValueError(f"Could not read image {path}")
    gray = preprocess_gray(img, size=size, use_clahe=use_clahe)
    feat = featurize_gray(gray, hog)[None, :]

    pred = int(model.predict(feat)[0])
    label = classes[pred]
    score = float(model.decision_function(feat)[0])
    if verbose:
        print(f"Prediction: {label}")
        print(f"Decision score: {score:.4f} (negative→{classes[0]}, positive→{classes[1]})")
    return label, score

def predict_dir(folder, bundle=BUNDLE_PATH):
    folder = Path(folder)
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    paths = [p for p in folder.rglob("*") if p.suffix in exts]
    if not paths:
        print("No images found.")
        return
    for p in paths:
        try:
            label, score = predict_image(str(p), bundle=bundle, verbose=False)
            print(f"{p.name:28s} -> {label:3s}  score={score:+.3f}")
        except Exception as e:
            print(f"{p.name:28s} -> ERROR: {e}")

def debug_image(path, bundle=BUNDLE_PATH):
    print(f"Debugging {path}")
    b, hog = _load_bundle(bundle)
    model, classes = b["model"], b["classes"]
    size, use_clahe = tuple(b["img_size"]), b.get("use_clahe", USE_CLAHE)
    img = safe_imread(path)
    if img is None: raise ValueError(f"Could not read {path}")

    def score_of(im):
        gray = preprocess_gray(im, size=size, use_clahe=use_clahe)
        feat = featurize_gray(gray, hog)[None, :]
        pred = int(model.predict(feat)[0])
        score = float(model.decision_function(feat)[0])
        return classes[pred], score

    lbl, sc = score_of(img)
    print(f"ORIG -> {lbl:3s}  score={sc:+.3f}  (neg→{classes[0]}, pos→{classes[1]})")
    lbl_f, sc_f = score_of(cv2.flip(img, 1))
    print(f"FLIP -> {lbl_f:3s}  score={sc_f:+.3f}")

# ---------- Live camera ----------
def live_camera(bundle=BUNDLE_PATH, cam_index=0, mirror=True, win_name="Cat/Dog Live"):
    """

    Open a webcam stream, classify each frame with the saved HOG+SVM, and overlay the label + score. Press:
    q = quit
    p = pause/resume (freeze-frame)
    s = save current frame to ./frame_YYYYmmdd_HHMMSS.jpg
    """

    b, hog = _load_bundle(bundle)
    model, classes = b["model"], b["classes"]
    size, use_clahe = tuple(b["img_size"]), b.get("use_clahe", USE_CLAHE)

    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise ValueError(f"Could not open camera {cam_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    paused = false
    fps_t0 = time.time()
    fps_hist = deque(maxlen=30)

    def predict_frame(frame_bgr):
        """Return (label, score) for a BGR frame."""
        gray = preprocess_gray(frame_bgr, size=size, use_clahe=use_clahe)
        feat = featurize_gray(gray, hog)[None, :]
        pred = int(model.predict(feat)[0])
        score = float(model.decision_function(feat)[0])
        return classes[pred], score

    GREEN = (0, 200, 0)
    RED = (0, 0, 200)
    WHITE = (240, 240, 240)
    BLACK = (0, 0, 0)

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                print("Frame grab failed; retrying...")
                continue

            if mirror:
                frame = cv2.flip(frame, 1)

            label, score = predict_frame(frame)
            color = GREEN if score >= 0 else RED
            text = f"{label} score={score:+.3f}"
            cv2.putText(frame, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, BLACK, 3, cv2.LINE_AA)
            cv2.putText(frame, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

            conf = 1.0 - np.exp(-min(5.0, abs(score)))
            bar_w = int(300 * conf)
            cv2.rectangle(frame, (10, 45), (10 + 300, 65), (60, 60, 60), 1)
            cv2.rectangle(frame, (10, 45), (10 + bar_w, 65), color, -1)

            t1 = time.time()
            fps_hist.append(1.0 / max(1e-6, (t1 - fps_t0)))
            fps_t0 = t1
            fps = sum(fps_hist) / max(1, len(fps_hist))
            cv2.putText(frame, f"FPS: {fps:4.1f}", (12, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)

            cv2.putText(frame, "q: quit   p: pause   s: save", (12, frame.shape[0]-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)

            cv2.imshow(win_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = not paused
        elif key == ord("s"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"frame_{ts}.jpg"

            if not paused:
                ok, frame_to_save = cap.read()
                if ok and mirror:
                    frame_to_save = cv2.flip(frame_to_save, 1)
            else:
                pass
            try:
                img_to_write = locals().get('frame', locals().get('frame_to_save', None))
                if img_to_write is not None:
                    cv2.imwrite(fname, img_to_write)
                    print(f"Saved frame {fname}")
            except Exception as e:
                print(f"Save failed: {e}")

    cap.release()
    cv2.destroyAllWindows()


# ---------- CLI ----------
if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("Usage:\n"
              "  python recognition.py --train\n"
              "  python recognition.py --predict path/to/image.jpg\n"
              "  python recognition.py --predict-dir path/to/folder\n"
              "  python recognition.py --debug path/to/image.jpg\n"
              "  python recognition.py --camera [index]")
        sys.exit(0)

    cmd = args[0]
    if cmd == "--train":
        train_and_save()
    elif cmd == "--predict" and len(args) == 2:
        predict_image(args[1])
    elif cmd == "--predict-dir" and len(args) == 2:
        predict_dir(args[1])
    elif cmd == "--debug" and len(args) == 2:
        debug_image(args[1])
    elif cmd == "--camera":
        idx = int(args[1]) if len(args) >= 2 else 0
        live_camera(cam_index=idx)
    else:
        print("Run without args for help")
