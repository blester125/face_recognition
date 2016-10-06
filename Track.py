from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import Faces
import Draw

def main():
	fc = Faces.Face_Classifier()
	print("1. Classify faces from the webcam")
	print("2. Classify faces from an image")
	print("3. Add training data with the webcam")
	print("4. Special Features")
	user_input = raw_input("Please make a selection: ")
	if int(user_input) == 1 or int(user_input) == 3:
		train = False
		label = None
		save = False
		if int(user_input) == 1:
			save_input = raw_input("Do you want to save the video? (y/n) ")
			if save_input == "y":
				save = True
		elif int(user_input) == 3:
			train = True
			label = raw_input("What is the label for this data? ")
		classify_from_camera(fc, train, label, save)
	elif int(user_input) == 2:
		filename = raw_input("Please enter a filename: ")
		classify_from_disk(fc, filename)
	elif int(user_input) == 4:
		print("1: Visualize embeddings (under construction)")
		print("2: Display Landmarks")
		print("3: Add Mustache")
		print("4: Both 2 and 3")
		user_input = raw_input("Please make a selection: ")
		save = False
		if int(user_input) != 1:
			save_input = raw_input("Do you want to save the video? (y/n) ")
			if save_input == "y":
				save = True
		if int(user_input) == 2:
			display_feature(fc, landmarks=True, save=save)
		elif int(user_input) == 3:
			display_feature(fc, mustache=True, save=save)
		elif int(user_input) == 4:
			display_feature(fc, landmarks=True, mustache=True, save=save)
	else:
		print("No valid selection. Exiting.")


def classify_from_disk(fc, filename):
	image = mpimg.imread(filename)
	faces = fc.classify_faces(image, True)
	for face in faces:
		fc.draw_face(image, face.get_rectangle(), face.get_color())
		fc.drawer.draw_label(image, face.get_label(), face.get_rectangle(), face.get_color()) 
	plt.imshow(image)
	plt.show()

def classify_from_camera(fc, train, label, save=False):
	video_cap = cv2.VideoCapture(0)
	detected_faces = []
	##
	# Saving
	if save == True:
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
	i = 0
	count = 0
	while True:
		ret, frame = video_cap.read()
		if i % 5 == 0:
			i = 0
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			if train == True:
				if count == 20:
					fc.train()
					break
				frame = fc.add_training_face(frame, label)
				count += 1
			else:
				detected_faces = fc.classify_faces(frame)
			frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
		for face in detected_faces:
			fc.draw_face(frame, face.get_rectangle(), face.get_color())
			fc.drawer.draw_label(
							frame, 
							face.get_label(), 
							face.get_rectangle(), 
							face.get_color())
		cv2.imshow('Video', frame)
		##
		# Saving
		if save == True:
			out.write(frame)
		i += 1
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	video_cap.release()
	##
	# Saving
	if save == True:
		out.release()
	cv2.destroyAllWindows()

def display_feature(fc, landmarks=False, mustache=False, save=False):
	video_cap = cv2.VideoCapture(0)
	detected_faces = []
	##
	# Saving
	if save == True:
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
	i = 0
	while True:
		ret, frame = video_cap.read()
		if i % 5 == 0:
			i = 0
			detected_faces = fc.face_detector.detect_faces(frame)
		for face in detected_faces:
			points = fc.face_detector.find_landmarks(frame, face)
			if landmarks == True:
				Draw.draw_points(frame, points, (0, 0, 0))
			if mustache == True:
				Draw.add_mustache(frame, points)

		cv2.imshow('Video', frame)
		##
		# Saving
		if save == True:
			out.write(frame)
		i += 1
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	video_cap.release()
	##
	# Saving
	if save == True:
		out.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
 	main()
