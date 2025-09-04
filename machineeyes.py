import cv2
from skimage.feature import hog
import matplotlib.pyplot as plt
from recognition import IMG_SIZE, preprocess_gray

img = cv2.imread("bella.jpg")
gray = preprocess_gray(img)  # matches training preprocessing
feat, hog_image = hog(
    gray,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    visualize=True,
    feature_vector=True
)

plt.imshow(hog_image, cmap="gray")
plt.title("HOG Visualization")
plt.show()
