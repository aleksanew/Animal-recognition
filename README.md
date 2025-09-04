🐾 Animal Recognition — Cats vs Dogs (Flask + OpenCV + scikit-learn)

This project is a local web application that recognizes whether an image contains a Cat or a Dog.

It uses:

HOG features (OpenCV)

Linear SVM classifier (scikit-learn)

A simple Flask web app with HTML/JS/CSS frontend

You can either upload an image or use your webcam directly from the browser.

🚀 Quick Start
1. Clone the repository
git clone https://github.com/aleksanew/Animal-recognition.git
2. cd Animal-recognition

2. Create a virtual environment
python -m venv .venv

3. Install dependencies
pip install -r requirements.txt

4. Download the pretrained model

This repo does not include the training dataset.
Instead, use the provided downloader to fetch the trained model from GitHub Releases:

python get_model.py


This will place catdog_hog_svm.joblib in your project folder.

5. Run the web app
python app.py


Then open http://127.0.0.1:5000
 in your browser.

🌐 Features

Upload an image → get classification + annotated preview

Use your webcam (browser getUserMedia) → click Start camera, then Capture & Predict

Stop camera button releases webcam properly

Annotated results show predicted label and decision score

💡 Tip: Decision scores

Positive → Dog

Negative → Cat

🔄 Optional: Retrain the Model

If you want to retrain:

Download the Microsoft Cats vs Dogs (PetImages) dataset manually and unzip into:

PetImages/Cat/*.jpg
PetImages/Dog/*.jpg


(⚠️ The dataset is ~1.25 GB, not included in this repo).

Train:

python recognition.py --train


This will produce a new catdog_hog_svm.joblib which app.py will load automatically.

📦 API Endpoints

POST /predict_upload
Input: multipart form with file=<image>
Output JSON:

{ "label": "Cat", "score": -1.234, "image": "data:image/jpeg;base64,..." }


POST /predict_snapshot
Input: JSON with "image": "data:image/jpeg;base64,..."
Output: same format as above

🛠️ Requirements

Python 3.9–3.12

Dependencies in requirements.txt:

Flask

OpenCV (opencv-python)

NumPy

scikit-learn

joblib

Install with:

pip install -r requirements.txt

🧪 Testing the App

Upload a cat or dog image (.jpg, .png)

Observe the predicted label + score overlay

Start your webcam, hold up an image/object, and press Capture & Predict

If you want to verify training:

python recognition.py --debug path/to/test.jpg


This prints predicted label + raw decision score.

🤝 Contributing

Pull requests and suggestions are welcome!

Use issues
 to report bugs

Fork the repo and submit PRs for new features
