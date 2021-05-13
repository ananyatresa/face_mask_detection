from sklearn.cluster import DBSCAN
import numpy as np
import pickle
import cv2
import os
from imutils import paths
import face_recognition
import shutil

def cluster_faces():
    print("LOADING ENCODINGS")
    data = pickle.loads(
        open(r"encodings.pickle", "rb").read())
    data = np.array(data)
    encodings = [d["encoding"] for d in data]

    print("clustering...")
    clt = DBSCAN(metric="euclidean", n_jobs=-1)
    clt.fit(encodings)
    # determine the total number of unique faces found in the dataset
    labelIDs = np.unique(clt.labels_)
    numUniqueFaces = len(np.where(labelIDs > -1)[0])
    print("[INFO] # unique faces: {}".format(numUniqueFaces))

    # loop over the unique face integers
    for labelID in labelIDs:
        # find all indexes into the `data` array that belong to the
        # current label ID, then randomly sample a maximum of 25 indexes
        # from the set
        print("[INFO] faces for face ID: {}".format(labelID))
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
            serial = 0000
            finalfacestore = r"final_face_store\face_" + str(
                serial + i) + ".jpg"
        cv2.imwrite(finalfacestore, faces[0])

        # create a montage using 96x96 "tiles" with 5 rows and 5 columns
        # montage = build_montages(faces, (96, 96), (5, 5))[0]
        # show the output montage
        # title = "Face ID #{}".format(labelID)
        # title = "Unknown Faces" if labelID == -1 else title
        # cv2.imshow(title, montage)
        cv2.waitKey(0)

def encode_faces():
    dataset = r"face-store"

    encoding_path = r"encodings.pickle"

    # grab the paths to the input images in our dataset, then initialize
    # out data list (which we'll soon populate)
    print("[INFO] quantifying faces...")
    imagePaths = list(paths.list_images(dataset))
    data = []

    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # load the input image and convert it from RGB (OpenCV ordering)
        # to dlib ordering (RGB)
        print("[INFO] processing image {}/{}".format(i + 1,
                                                     len(imagePaths)))
        print(imagePath)
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")
        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)
        # build a dictionary of the image path, bounding box location,
        # and facial encodings for the current image
        d = [{"imagePath": imagePath, "loc": box, "encoding": enc}
             for (box, enc) in zip(boxes, encodings)]
        data.extend(d)

    # dump the facial encodings data to disk
    print("[INFO] serializing encodings...")
    f = open(encoding_path, "wb")
    f.write(pickle.dumps(data))
    f.close()

def delete_faces():
    folder = r"face-store"
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def generate_faces():
    encode_faces()
    cluster_faces()
    delete_faces()

