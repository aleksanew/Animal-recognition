from flask import Flask, request, jsonify, render_template
import numpy as np, cv2, base64
from recognition import _load_bundle, preprocess_gray, featurize_gray, USE_CLAHE

# ---- Load model & HOG once ----
b, hog = _load_bundle()
model, classes = b["model"], b["classes"]
size      = tuple(b["img_size"])
use_clahe = b.get("use_clahe", USE_CLAHE)

def predict_bgr(img_bgr):
    gray = preprocess_gray(img_bgr, size=size, use_clahe=use_clahe)
    feat = featurize_gray(gray, hog)[None, :]
    pred = int(model.predict(feat)[0])
    score = float(model.decision_function(feat)[0])
    return classes[pred], score

def annotate(img_bgr, label, score):
    out = img_bgr.copy()
    color = (0,200,0) if score >= 0 else (0,0,200)
    text = f"{label}  score={score:+.3f}"
    cv2.putText(out, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return out

def bgr_from_file(file_storage):
    data = file_storage.read()
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

def bgr_from_base64(data_url):
    b64 = data_url.split(",", 1)[1] if "," in data_url else data_url
    raw = base64.b64decode(b64)
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)

def bgr_to_data_url(img_bgr):
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.post("/predict_upload")
def predict_upload():
    if "file" not in request.files:
        return jsonify(error="no file"), 400
    img = bgr_from_file(request.files["file"])
    if img is None:
        return jsonify(error="bad image"), 400
    label, score = predict_bgr(img)
    annotated = annotate(img, label, score)
    return jsonify(label=label, score=score, image=bgr_to_data_url(annotated))

@app.post("/predict_snapshot")
def predict_snapshot():
    data = request.get_json(silent=True) or {}
    img = bgr_from_base64(data.get("image", ""))
    if img is None:
        return jsonify(error="bad image"), 400
    label, score = predict_bgr(img)
    annotated = annotate(img, label, score)
    return jsonify(label=label, score=score, image=bgr_to_data_url(annotated))

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
