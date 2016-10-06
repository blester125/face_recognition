from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import Detector
import Classifier
import Draw

import cv2
import os

import matplotlib.pyplot as plt
import scipy.misc

class Result():
	def __init__(self, rectangle=None, label=None, color=(0, 0, 0)):
		self.rectangle = rectangle
		self.label = label
		self.color = color

	def get_rectangle(self):
		return self.rectangle

	def get_label(self):
		return self.label

	def get_color(self):
		return self.color

class Face_Classifier():
	def __init__(
			self,
			predictor_path="shape_predictor_68_face_landmarks.dat",
			network_path="nn4.small2.v1.t7",
			imgDim=96,
			dataPath="data.csv",
			modelPath="model.pkl",
			LEPath="LabelEncoder.pkl",
			lineThickness=2,
			font=cv2.FONT_HERSHEY_PLAIN,
			fontScale=2.0,
			textThickness=2,
			spacer=2):
		self.face_detector = Detector.Detector(
										predictor_path, 
										network_path, 
										imgDim)
		self.face_classifier = Classifier.Classifier(
											dataPath,
											modelPath,
											LEPath)
		self.drawer = Draw.Draw(
							lineThickness,
							font,
							fontScale,
							textThickness,
							spacer)

	def classify_faces(self, image, RGB=False):
		faces = self.face_detector.detect_faces(image)
		detected_faces = []
		for face in faces:
			points = self.face_detector.find_landmarks(image, face)
			img = self.face_detector.align(image, points)
			embedding = self.face_detector.get_embedding(img)
			#label = self.face_classifier.infer([embedding])
			#label = self.face_classifier.get_label(label)[0]
			label = self.face_classifier.classify_face([embedding])
			color = self.face_classifier.get_color(label, RGB)
			face = Result(face, label, color)
			detected_faces.append(face)
		return detected_faces

	def draw_face(self, image, face, color):
		self.drawer.draw_face(image, face, color)

	def draw_text(self, image, label, face, color):
		self.drawer.draw_label(image, label, face, color)

	##
	# Add so that the image is saved to a folder of trian/{label}
	def add_training_face(self, image, label):
		path = os.path.join(os.getcwd(), 'train', label)
		if not os.path.exists(path):
			os.mkdir(path)
		faces = self.face_detector.detect_faces(image)
		if len(faces) == 1:
			points = self.face_detector.find_landmarks(image, faces[0])
			img = self.face_detector.align(image, points)
			embedding = self.face_detector.get_embedding(img)
			self.face_classifier.add_sample([embedding], [label])
			files = os.listdir(path)
			filename = 'image-' + str(len(files) + 1) + ".jpg"
			filename = os.path.join(path, filename)
			scipy.misc.imsave(filename, image)
			self.drawer.draw_face(image, faces[0], (0, 0, 0))
			return image

	def train(self):
		self.face_classifier.train()