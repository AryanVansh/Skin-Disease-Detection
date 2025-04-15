import cv2

img = cv2.imread("C:/Users/User/OneDrive/Desktop/vs/Skin/data_total/all_images/ISIC_0033319.jpg")
if img is None:
    print("Image is unreadable or corrupted.")
else:
    print("Image loaded successfully.")
