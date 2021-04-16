import tensorflow
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed

def get_embedding(model, face_pixels):
	face_pixels = face_pixels.astype('float32')
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	samples = expand_dims(face_pixels, axis=0)
	yhat = model.predict(samples)
	return yhat[0]

if __name__ == "__main__":
  data = load('files\\8th_sem_data.npz')
  trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
  print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)
  model = tensorflow.keras.models.load_model('model\\facenet_keras.h5')
  newTrainX = list()
  for face_pixels in trainX:
    embedding = get_embedding(model, face_pixels)
    newTrainX.append(embedding)
  newTrainX = asarray(newTrainX)
  print(newTrainX.shape)
  newTestX = list()
  for face_pixels in testX:
    embedding = get_embedding(model, face_pixels)
    newTestX.append(embedding)
  newTestX = asarray(newTestX)
  savez_compressed('files\\8th_sem_embeddings.npz', newTrainX, trainy, newTestX, testy)