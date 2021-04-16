import os
from os.path import isdir
from PIL import Image
from matplotlib import pyplot
from numpy import savez_compressed
from numpy import asarray
from mtcnn.mtcnn import MTCNN

def extract_face(filename, required_size=(160, 160)):
	image = Image.open(filename)
	image = image.convert('RGB')
	pixels = asarray(image)
	detector = MTCNN()
	results = detector.detect_faces(pixels)
	x1, y1, width, height = results[0]['box']
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	face = pixels[y1:y2, x1:x2]
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

def load_faces(directory):
	faces = list()
	for filename in os.listdir(directory):
		path = os.path.join(directory, filename)
		face = extract_face(path)
		faces.append(face)
	return faces

def load_dataset(directory):
	X, y = list(), list()
	for subdir in os.listdir(directory):
		path = os.path.join(directory, subdir)
		if not isdir(path):
			continue
		faces = load_faces(path)
		labels = [subdir for _ in range(len(faces))]
		print('>loaded %d examples for class: %s' % (len(faces), subdir))
		X.extend(faces)
		y.extend(labels)
	return asarray(X), asarray(y)

if __name__ == "__main__":
  trainX, trainy = load_dataset('data\\train')
  testX, testy = load_dataset('data\\val')
  savez_compressed('8th_sem_data.npz', trainX, trainy, testX, testy)