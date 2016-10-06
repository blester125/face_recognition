from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import cv2
import openface
import os
import dlib

import pandas as pd
from pandas import DataFrame

import Draw
import Detector

def create_dataset(path, cls, detector, index):
	data = []
	label = []

	for file in os.listdir(path):
		filename = os.path.join(path, file)
		#print(filename)
		image = cv2.imread(filename)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		detected_faces = detector.detect_faces(image)
		if len(detected_faces) != 1:
			print("No/Many faces found in " + path + " " + file)
			continue
		for face in detected_faces:
			points = detector.find_landmarks(image, face)
			img = detector.align(image, points)
			# print("out/" + filename)
			# cv2.imwrite("out/" + filename, img)
			rep = detector.get_embedding(img)
			data.append(rep)
			label.append(cls)

	data = np.array(data)
	label = np.array(label)
	df = DataFrame(data=data, index=np.arange(index, index+data.shape[0]))
	df['Label'] = label
	return df, data.shape[0]
			
def save_dataset(path, dataframes):
	df = pd.concat(dataframes)
	df.to_csv(path)

if __name__ == "__main__":
	face_detector = Detector.Detector(
							"shape_predictor_68_face_landmarks.dat",
							"nn4.small2.v1.t7",
							96)
	dataframes = []
	length = 0
	for directory in os.listdir("train"):
		path = os.path.join("train", directory)
		df, length = create_dataset(path, directory, face_detector, length)
		dataframes.append(df)
	save_dataset("data.csv", dataframes)
