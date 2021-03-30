import time
import cv2
webcam = cv2.VideoCapture(0) #Use camera 0

# We load the xml file
#classifier = cv2.CascadeClassifier(r"C:\Users\tauru\OneDrive\Desktop\project\face-mask-detector\haar\haarcascade_frontalface_default    ")
classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("[INFO] starting video stream...")
# vs = VideoStream(usePiCamera=True).start()


while True:
    _, vs = webcam.read()
    time.sleep(2.0)
    total = 0
    gray = cv2.cvtColor(vs, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = classifier.detectMultiScale(gray, 1.1, 4)
    # loop over the face detections and draw them on the frame

    for (x, y, w, h) in faces:
        cv2.rectangle(vs, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Save just the rectangle faces in SubRecFaces
        sub_face = vs[y:y + h, x:x + w]
        FaceFileName = r"C:\Users\tauru\OneDrive\Desktop\project\face-mask-detector\face-store\face_" + str(y) + ".jpg"
        cv2.imwrite(FaceFileName, sub_face)
    # Show the image
    cv2.imshow('face capture', vs)
    key = cv2.waitKey(10)

    #enter esc to exit
    if key == 27:
        break
