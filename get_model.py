import os, sys, pathlib, urllib.request

MODEL = "catdog_hog_svm.joblib"
URL   = "https://github.com/aleksanew/Animal-recognition/releases/download/v1.0/catdog_hog_svm.joblib"  # <-- your release URL

def main():
    if pathlib.Path(MODEL).exists():
        print(f"{MODEL} already present.")
        return
    print(f"Downloading {MODEL} ...")
    try:
        urllib.request.urlretrieve(URL, MODEL)
        print("Done.")
    except Exception as e:
        print("Download failed:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
