from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os



dataset=r"C:\Users\tauru\OneDrive\Desktop\project\face_mask_detector\face-store"

encoding_path=r"C:\Users\tauru\OneDrive\Desktop\project\face_mask_detector\encodings.pickle"


# grab the paths to the input images in our dataset, then initialize
# out data list (which we'll soon populate)
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(dataset))
data = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# load the input image and convert it from RGB (OpenCV ordering)
	# to dlib ordering (RGB)
    print("processing image {}/{}".format(i + 1,
    	len(imagePaths)))
    print(imagePath)
    image = cv2.imread(imagePath)
    rgb=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model="hog")
    # compute the facial embedding for the face
    encodings = face_recognition.face_encodings(rgb, boxes)
    # build a dictionary of the image path, bounding box location,
    # and facial encodings for the current image
    d = [{"imagePath": imagePath, "loc": box, "encoding": enc}
         for (box, enc) in zip(boxes, encodings)]
    data.extend(d)

# dump the facial encodings data to disk
print("serializing encodings...")
f = open(encoding_path, "wb")
f.write(pickle.dumps(data))
f.close()
