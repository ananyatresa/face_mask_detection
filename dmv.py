from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from tkinter import *
import tkinter.font as font
import numpy as np
import imutils
import time
import cv2
import os
import PIL
from PIL import Image,ImageTk


width, height = 200, 300
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

def switch(m):
    if b1["text"] == "START":
        l1["height"] = 300
        l1["width"] = 400
        l1.grid(row=1, column=0) 
        b1["bg"] = "red"
        b1["text"] = "STOP"
        show_frame()
    else:
        print ("bhe")
        l1["bg"] = "black", 
        l1["fg"] = "purple"
        l1["text"] = "Press START for video streaming"
        l1.grid(row=1, column=0) 
        b1["bg"] = "blue"
        b1["text"] = "START"
        close_frame()

gui = Tk()
gui.title("Face Mask Detection")
# set window size
gui.geometry("550x550")

# set window color
gui.configure(bg='white')
gui.columnconfigure(0, minsize=250)
gui.rowconfigure([2, 2], minsize=100)

path=r"D:\ANANYA\College\FINAL YEAR Project\Face mask model\git copy\face_mask_detection\face-store"
def open(m):
    os.startfile(path, 'open')

myFont = font.Font(family='Arial', size=16, weight='bold')

lbl_greeting = Label(text=" WELCOME TO FACE MASK DETECTION ", fg="purple1", bg="white")
lbl_greeting.grid(row=0, column=0, padx=10, pady=10)
lbl_greeting["font"] = myFont

l1 = Label(gui, text="Press START for Video Streaming", fg="purple", bg="black", height=21, width=50)
l1.grid(row=1, column=0) 

frame=LabelFrame(gui, bg="white", padx=30, pady=30)
frame.grid(row=2,column=0, padx=10, pady=10)

#Buttons
b1 = Button(frame, text="START", width=10, height=2, fg="white", bg="blue", relief="ridge")
b1.grid(row=2, column=0, sticky="S", padx=10, pady=10)
b1.bind("<Button-1>", switch)

b2 = Button(frame, text="GENERATE FACES", width=15, height=2, fg="white", bg="green", relief="ridge")
b2.grid(row=2, column=1, sticky="S",padx=10, pady=10)
b2.bind("<Button-1>", open)

def close_frame():
    cap.release()
    #l2=Label(gui,bg="green", height=21, width=50)


def detect_and_predict_mask(frame, faceNet, maskNet):

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                    (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]
        
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)

            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))



    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)



print("loading face detector model...")
prototxtPath = r"D:\ANANYA\College\FINAL YEAR Project\Face mask model\git copy\face_mask_detection\deploy.prototxt"
weightsPath = r"D:\ANANYA\College\FINAL YEAR Project\Face mask model\git copy\face_mask_detection\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("loading face mask detector model...")
maskNet = load_model(r"D:\ANANYA\College\FINAL YEAR Project\Face mask model\git copy\face_mask_detection\mask_detector.model")

print("starting video stream...")
#vs = VideoStream(src=0).start()
#time.sleep(2.0)
# loop over the frames from the video stream
def show_frame():
    #frame = vs.read()
    
        
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=400)

    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
    
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)


        if mask < withoutMask:
            sub_face = frame[startY:endY, startX:endX]
            FaceFileName = r"D:\ANANYA\College\FINAL YEAR Project\Face mask model\git copy\face_mask_detection\face-store\face_" + str(startY) + ".jpg"
            cv2.imwrite(FaceFileName, sub_face)

    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = PIL.Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    l1.imgtk = imgtk
    l1.configure(image=imgtk)
    l1.after(10, show_frame)

gui.mainloop()

        

        # cv2.imshow("Frame", frame)
        # key = cv2.waitKey(1) & 0xFF
        
        # # press q to exit
        # if key == ord("q"):
        #     break
        


    # cv2.destroyAllWindows()
    # vs.stop()

























