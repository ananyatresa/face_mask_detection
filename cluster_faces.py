from sklearn.cluster import DBSCAN
from imutils import build_montages
import numpy as np
import pickle
import cv2
import os

# load the serialized face encodings + bounding box locations from
# disk, then extract the set of encodings to so we can cluster on
# them
print("LOADING ENCODINGS")
data = pickle.loads(open(r"C:\Users\tauru\OneDrive\Desktop\project\face_mask_detector\encodings.pickle", "rb").read())
data = np.array(data)
encodings = [d["encoding"] for d in data]

print("clustering...")
clt = DBSCAN(metric="euclidean", n_jobs=-1)
clt.fit(encodings)
# determine the total number of unique faces found in the dataset
labelIDs = np.unique(clt.labels_)
numUniqueFaces = len(np.where(labelIDs > -1)[0])
print("# unique faces: {}".format(numUniqueFaces))

# loop over the unique face integers
for labelID in labelIDs:
    # find all indexes into the `data` array that belong to the
    # current label ID, then randomly sample a maximum of 25 indexes
    # from the set
    print("faces for face ID: {}".format(labelID))
    idxs = np.where(clt.labels_ == labelID)[0]
    idxs = np.random.choice(idxs, size=min(25, len(idxs)), replace=False)

    # initialize the list of faces to include in the montage
    faces = []
    for i in idxs:
        # load the input image and extract the face ROI
        image = cv2.imread(data[i]["imagePath"])
        (top, right, bottom, left) = data[i]["loc"]
        face = image[top:bottom, left:right]
        # force resize the face ROI to 96x96 and then add it to the
        # faces montage list
        face = cv2.resize(face, (96, 96))
        faces.append(face)
        serial= 0
        finalfacestore=r"C:\Users\tauru\OneDrive\Desktop\project\face_mask_detector\final_face_store\face_000"+ str(serial+i) +".jpg"
    cv2.imwrite(finalfacestore,faces[0])

    # create a montage using 96x96 "tiles" with 5 rows and 5 columns
    #montage = build_montages(faces, (96, 96), (5, 5))[0]
    # show the output montage
    #title = "Face ID #{}".format(labelID)
    #title = "Unknown Faces" if labelID == -1 else title
    #cv2.imshow(title, montage)
    cv2.waitKey(0)
