from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np 
import cv2
import math

from scipy.misc import imresize
import matplotlib.pyplot as plt

class Draw():
	def __init__(
			self,
			line_thickness=2,
			font=cv2.FONT_HERSHEY_PLAIN,
			font_scale=2.0,
			text_thickness=3,
			spacer=2):
		self.line_thickness = line_thickness
		self.font = font
		self.font_scale = font_scale
		self.text_thickness = text_thickness
		self.spacer = spacer

	def draw_face(self, image, face_rect, color=(0, 255, 0)):
		draw_box(image, face_rect, color, self.line_thickness)

	def draw_label(self, image, label, face_rect, color=(0, 255, 0)):
		height = get_text_height(label, self.font, self.font_scale, self.text_thickness)
		offset = self.line_thickness + height + self.spacer
		write_text(
				image, 
				label, 
				(face_rect.left(), face_rect.bottom() + offset), 
				self.font, 
				self.font_scale, 
				color, 
				self.text_thickness)


def draw_box(image, rectangle, color=(0, 255, 0), thickness=2):
	cv2.rectangle(
			image,
			(rectangle.left(), rectangle.top()),
			(rectangle.right(), rectangle.bottom()),
			color,
			thickness
		)
	return image

def write_text(
		image, 
		text="", 
		point=(0,0), 
		font=cv2.FONT_HERSHEY_PLAIN, 
		font_scale=1.0,
		color=(0,0,0),
		thickness=2
	):
	cv2.putText(
			image,
			text,
			point,
			font,
			font_scale,
			color,
			thickness
		)
	return image

def get_text_size(
		text, 
		font=cv2.FONT_HERSHEY_PLAIN, 
		font_scale=1.0, 
		thickness=2
	):
	return cv2.getTextSize(text, font, font_scale, thickness)

def get_text_height(
		text, 
		font=cv2.FONT_HERSHEY_PLAIN, 
		font_scale=1.0, 
		thickness=2
	):
	size = get_text_size(text, font, font_scale, thickness)
	return size[0][1]

def draw_points(image, points=[], color=(0, 255, 0)):
	for point in points:
		cv2.circle(image, point, 2, color, -2)
	return image

def get_size_of_mouth(points):
	size = points[48][0] - points[54][0]
	if size < 0:
		size = size * -1
	return size

def get_size_between_nose_mouth(points):
	size = points[33][1] - points[51][1]
	if size < 0:
		size = size * -1
	return size

def resize(img, height, width):
	img = imresize(img, (height,width))
	return img

def add_mustache(image, points):
	width = get_size_of_mouth(points)
	height = get_size_between_nose_mouth(points)
	mustache = cv2.imread("mustache.png")
	scaled = resize(mustache, height, width)
	offseth = 0
	offsetw = 0
	#if scaled.shape[0] % 2 != 0:
	#	offseth = 1
	if scaled.shape[1] % 2 != 0:
		offsetw = 1
	image[
		(points[51][1] - int(height)+offseth):(points[51][1]),
		(points[51][0] - int(width/2)):(points[51][0] + int(width/2) + offsetw),
		:
	] = scaled
	return image

def add_eyes(image, points):
	LEFT_EYE = [36, 37, 38, 39, 40, 41]
	RIGHT_EYE = [42, 43, 44, 45, 46, 47]
	pts = [points[x] for x in LEFT_EYE]
	pts = np.array(pts)
	cv2.fillPoly(image, [pts], (255, 255, 255))
	pts = [points[x] for x in RIGHT_EYE]
	pts = np.array(pts)
	cv2.fillPoly(image, [pts], (255, 255, 255))
	return image

def add_blood(image, points):
	LEFT_EYE = [48, 60, 7]
	RIGHT_EYE = [54, 64, 9]
	pts = [points[x] for x in LEFT_EYE]
	pts = np.array(pts)
	cv2.fillPoly(image, [pts], (0, 0, 255))
	#pts = [points[x] for x in RIGHT_EYE]
	#pts = np.array(pts)
	#cv2.fillPoly(image, [pts], (0, 0, 255))
	return image

def add_fangs(image, points):
	LEFT_EYE = [49, 50, 59]
	RIGHT_EYE = [53, 52, 55]
	pts = [points[x] for x in LEFT_EYE]
	pts = np.array(pts)
	cv2.fillPoly(image, [pts], (255, 255, 255))
	pts = [points[x] for x in RIGHT_EYE]
	pts = np.array(pts)
	cv2.fillPoly(image, [pts], (255, 255, 255))
	return image

def add_frank(image, points):
	width = int(points[16][0] - points[0][0])
	height = int(width * 400 / 640)
	frank = cv2.imread("frank.png")
	scaled = resize(frank, height, width)
	frank_range = (points[19][1] - (points[19][1] - height))
	image[(points[19][1] - height):points[19][1],points[0][0]:points[0][0]+width] = scaled
	return image

def add_bolts(image, points):
	hor = cv2.imread("hor.png")
	vert = cv2.imread("vert.png")
	shorter = points[11][1] - points[12][1]
	longer = int(shorter * 640 / 400)
	scaled_hor = resize(hor, shorter, longer)
	scaled_block = resize(vert, shorter, shorter)
	# Right
	image[(points[9][1] - shorter):points[9][1],points[11][0]:points[11][0]+longer] = scaled_hor
	image[(points[9][1] - shorter - shorter):(points[9][1] - shorter),points[11][0]+longer:points[11][0]+longer+shorter] = scaled_block
	image[(points[9][1] - shorter):(points[9][1]),points[11][0]+longer:points[11][0]+longer+shorter] = scaled_block
	image[(points[9][1]):(points[9][1] + shorter),points[11][0]+longer:points[11][0]+longer+shorter] = scaled_block
	# Left
	image[(points[7][1] - shorter):points[7][1],points[5][0]-longer:points[5][0]] = scaled_hor
	image[(points[7][1] - shorter - shorter):(points[7][1] - shorter),points[5][0]-longer-shorter:points[5][0]-longer] = scaled_block
	image[(points[7][1] - shorter):(points[7][1]),points[5][0]-longer-shorter:points[5][0]-longer] = scaled_block
	image[(points[7][1]):(points[7][1] + shorter),points[5][0]-longer-shorter:points[5][0]-longer] = scaled_block
	return image

