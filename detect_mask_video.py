# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from tkinter import *
import tkinter.font as font
import tkinter as tk
import numpy as np
import imutils
import time
import cv2
import os

def button1_clicked(m):

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
	'''
	ap = argparse.ArgumentParser()
	ap.add_argument("-f", "--face", type=str,
		default="face_detector",
		help="path to face detector model directory")
	ap.add_argument("-m", "--model", type=str,
		default="mask_detector.model",
		help="path to trained face mask detector model")
	ap.add_argument("-c", "--confidence", type=float, default=0.5,
		help="minimum probability to filter weak detections")
	args = vars(ap.parse_args())

	'''

	print("loading face detector model...")
	prototxtPath = r"deploy.prototxt"
	weightsPath = r"res10_300x300_ssd_iter_140000.caffemodel"
	faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

	# load the face mask detector model from disk
	print("loading face mask detector model...")
	maskNet = load_model("mask_detector.model")

	print("starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

	# loop over the frames from the video stream
	while True:

		frame = vs.read()
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

		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		#press q to exit
		if key == ord("q"):
			break


	cv2.destroyAllWindows()
	vs.stop()

gui = tk.Tk()
gui.title("Face Mask Detection")
# set window size
gui.geometry("400x300")

#set window color
gui.configure(bg='white')
gui.columnconfigure(0, minsize=250)
gui.rowconfigure([0,1], minsize=100)

myFont = font.Font(family='Arial', size=16, weight='bold')

lbl_greeting = tk.Label(text=" Welcome to Face Mask Detection :)) ", fg="purple1", bg="white")
lbl_greeting.grid(row=0, column=0, padx=10, pady=10)
lbl_greeting["font"]=myFont

btn_start = tk.Button(gui, text="Start", width=10, height=2, fg="white", bg="blue", relief="ridge")
btn_start.grid(row=1, column=0, sticky="S")
btn_start.bind("<Button-1>", button1_clicked)

gui.mainloop()