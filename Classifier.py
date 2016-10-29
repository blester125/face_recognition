from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import LabelEncoder

import cPickle as pickle

##
# Tune the Anomaly detector
class Classifier():
	def __init__(
			self, 
			path="data.csv", 
			modelpath="model.pkl", 
			le_path="LabelEncoder.pkl",
			anomaly_path="Anomaly.pkl"):
		##
		# Colors encoded in BGR format
		self.color_map = {
							"Brian": (204, 0, 102),
							"Amy": (255, 0, 0),
							"Walt": (0, 255, 0),
							"Vicki": (255, 255, 0),
							"Unknown": (0, 0, 255),
							"will": (45, 200, 150),
							"Chad": (255, 0, 0),
							"Jimmy": (233, 250, 254),
							#"Dave": (0, 255, 0)
						 }
		self.data = None
		self.labels = None
		self.datafile = path
		self.load_data(self.datafile)
		self.model_path = modelpath
		with open(self.model_path, "rb") as f:
			try:
				self.SVM = pickle.load(f)
			except:
				print("Unable to load model from", self.model_path)
				print("Be sure to train the model before use")
				self.SVM = None
		self.label_path = le_path
		with open(self.label_path, "rb") as f:
			try:
				self.LabelEncoder = pickle.load(f)
			except:
				print("Unable to load the Label Encoder from", self.label_path)
				print("Be sure to train the model before use")
				self.LabelEncoder = None
		if self.LabelEncoder is not None:
			try:
				self.labels_encoded = self.LabelEncoder.transform(self.labels)
			except ValueError:
				self.LabelEncoder.fit(self.labels)
				self.labels_encoded = self.LabelEncoder.transform(self.labels)
		self.anomaly_path = anomaly_path
		with open(self.anomaly_path, "rb") as f:
			try:
				self.anomaly_detectors = pickle.load(f)
			except:
				print("Unable to load the Anomaly Detectors from", self.anomaly_path)
				print("Be sure to train the model before use")
				self.anomaly_detectors = {}

	def load_data(self, path):
		try:
			df = pd.read_csv(path)
			self.labels = df['Label']
			self.labels = np.array(self.labels)
			self.data = df[range(1, df.shape[1] - 1)]
			self.data = np.array(self.data)
		except:
			print("Error loading data from", self.datafile)

	def add_sample(self, data, label):
		self.data = np.append(self.data, data, axis=0)
		self.labels = np.append(self.labels, label, axis=0)
		self.labels_encoded = self.LabelEncoder.fit_transform(self.labels)
		df = pd.DataFrame(data=self.data)
		df['Label'] = self.labels
		df.to_csv(self.datafile)

	def train(self):
		self.LabelEncoder = LabelEncoder()
		self.LabelEncoder.fit(self.labels)
		self.labels_encoded = self.LabelEncoder.transform(self.labels)
		self.SVM = SVC(C=1, kernel='linear')
		self.SVM.fit(self.data, self.labels_encoded)
		with open(self.model_path, "wb") as f:
			pickle.dump(self.SVM, f, -1)
		with open(self.label_path, "wb") as f:
			pickle.dump(self.LabelEncoder, f, -1)
		self.distinct_labels = set(self.labels)
		for label in self.distinct_labels:
			self.anomaly_detectors[label] = OneClassSVM()
			idx = np.where(self.labels == label)
			self.anomaly_detectors[label].fit(self.data[idx], self.labels[idx])
		with open(self.anomaly_path, "wb") as f:
			pickle.dump(self.anomaly_detectors, f, -1)

	def classify_face(self, sample):
		label_num = self.infer(sample)
		label = self.get_label(label_num)
		anomaly_label = self.check_anomaly(label[0], sample)
		#if anomaly_label[0] > 0:
		#	return label[0]
		#else:
		#	return "Unknown"
		return label[0]

	def infer(self, sample):
		assert self.SVM is not None
		return self.SVM.predict(sample)

	def get_label(self, label):
		assert self.LabelEncoder is not None
		return self.LabelEncoder.inverse_transform(label)

	def get_color(self, label, RGB=True):
		if label in self.color_map:
			if RGB:
				return self.color_map[label][::-1]
			return self.color_map[label]
		else:
			return (0, 0, 0)

	def check_anomaly(self, label, sample):
		if label in self.anomaly_detectors:
			return self.anomaly_detectors[label].predict(sample)

	def get_training_score(self):
		print(self.SVM.score(self.data, self.labels_encoded))

def classify_faces(image, detector, classifier):
	faces = detector.detect_faces(image)
	detected_faces = []
	for face in faces:
		points = detector.find_landmarks(image, face)
		img = detector.align(image, points)
		embedding = detector.get_embedding(img)
		label = classifier.classify_face([embedding])
		color = classifier.get_color(label)
		face = Result(face, label, color)
		detected_faces.append(face)
	return detected_faces

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

if __name__ == "__main__":
	import cv2
	import Detector
	import Draw
	import matplotlib.pyplot as plt
	drawer = Draw.Draw()
	#image = cv2.imread("Ferrell.jpg")
	image = cv2.imread("Freeman.jpg")
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	face_detector = Detector.Detector(
							"shape_predictor_68_face_landmarks.dat",
							"nn4.small2.v1.t7",
							96)
	svm = Classifier()
	svm.train()
	svm.get_training_score()
	detected_faces = classify_faces(image, face_detector, svm)
	for face in detected_faces:
		drawer.draw_face(image, face.get_rectangle(), face.get_color())
		drawer.draw_label(image, face.get_label(), face.get_rectangle(), face.get_color())
	plt.imshow(image)
	plt.show()
