from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from show_faces import generate_faces
from imutils.video import VideoStream
import tkinter.ttk as ttk
from tkinter import font
from PIL import ImageTk
from queue import Queue
from PIL import Image
import tkinter as tk
import numpy as np
import threading
import imutils
import time
import os
import sys
import cv2

#Creating class for tkinter and calling face mask detector through it
class App(tk.Frame):
    def __init__(self, parent, title):
        tk.Frame.__init__(self, parent)

        print("loading face detector model...")      
        self.is_running = False
        self.thread = None
        self.queue = Queue()
        self.photo = ImageTk.PhotoImage(Image.new("RGB", (400, 300), "white"))
        parent.wm_title(title)
        self.create_ui()
        self.grid(sticky=tk.NSEW)
        self.bind('<<MessageGenerated>>', self.on_next_frame)
        parent.geometry("550x550")
        parent.grid_rowconfigure([2,2], weight = 1)
        parent.grid_columnconfigure(0, weight = 1)       
        self.path=r"final_face_store"

#Creating tkinter GUI, labels, buttons
    def create_ui(self):

        self.lbl_greeting = ttk.Label(text=" WELCOME TO FACE MASK DETECTION ", foreground="purple", background="white")
        self.lbl_greeting.grid(row=0, column=0, padx=10, pady=10)
        self.myFont = font.Font(family='Arial', size=16, weight='bold')
        self.lbl_greeting["font"] = self.myFont
        self.lbl_capture = ttk.Label(text=" Press START to capture!! ", foreground="purple", background="white")
        self.lbl_capture.grid(row=1, column=0, padx=20, pady=1)       
        self.button_frame = tk.Frame(self, padx=20, pady=20)
        self.button_frame.grid(row=2,column=1, padx=5, pady=5)
        self.switch_button = tk.Button(self.button_frame, text="START", width=10, height=2, foreground="white", background="blue", relief="ridge", command=self.switch)
        self.switch_button.grid(row=2, column=0, sticky="S", padx=10, pady=10)
        self.generate_faces_button = tk.Button(self.button_frame, text="GENERATE FACES", width=15, height=2, foreground="white", background="green", relief="ridge", command=self.open)
        self.generate_faces_button.grid(row=2, column=1, sticky="S",padx=20, pady=10)

        self.view = tk.Label(self, image=self.photo)
        self.view.grid(row=1,column=1, padx=70, pady=30, sticky="e")

#Function for generating faces in GUI
    def open(self):
        generate_faces()
        os.startfile(self.path, 'open')

#Function for creating start-stop switch button in GUI
    def switch(self):
        if self.switch_button["text"] == "START":
            self.switch_button["background"] = "red"
            self.switch_button["text"] = "STOP"
            self.start()
        else:
            self.switch_button["background"] = "blue"
            self.switch_button["text"] = "START"
            self.stop()

#Function for start button in GUI, calls videoloop (face mask detector) function and initializes thread
    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.daemon = True
        self.thread.start()

#Function to stop/break from face mask detection loop
    def stop(self):
        self.is_running = False

#Function to loop over detections
    def detect_and_predict_mask(self, frame, faceNet, maskNet):

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

#Initializing webcam and loading model   
    def videoLoop(self):
        # No=0
        # cap = cv2.VideoCapture(No)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        width, height = 400, 300
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        prototxtPath = r"deploy.prototxt"
        weightsPath = r"res10_300x300_ssd_iter_140000.caffemodel"
        faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
        # load the face mask detector model from disk
        print("loading face mask detector model...")
        maskNet = load_model(r"mask_detector.model")
        print("starting video stream...")

        while self.is_running:
            
            _, frame = cap.read()
            frame = cv2.flip(frame, 1)
            frame = imutils.resize(frame, width=400)

            (locs, preds) = self.detect_and_predict_mask(frame, faceNet, maskNet)
    
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
                    FaceFileName = r"face-store\face_" + str(startY) + ".jpg"
                    cv2.imwrite(FaceFileName, sub_face)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            self.queue.put(image)
            self.event_generate('<<MessageGenerated>>')

#Function linking videoloop() to tkinter GUI() using queue
    def on_next_frame(self, eventargs):
        if not self.queue.empty():
            image = self.queue.get()
            image = Image.fromarray(image)
            self.photo = ImageTk.PhotoImage(image)
            self.view.configure(image=self.photo)


def main(args):
    gui = tk.Tk()
    app = App(gui, "Face Mask Detection Sytsem")
    gui.mainloop()

if __name__ == '__main__':
    sys.exit(main(sys.argv))
