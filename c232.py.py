rom numpy import loadtxt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
# load the dataset
dataset = loadtxt('diabetes_dataset.csv', delimiter=',')
# split into input (x) and output (y) variables
x = dataset[:,1 : 7]
y = dataset[:,8]
print("value of X are:", x)
print("value of Y are:", y)

# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(x, y, epochs=250, batch_size=100)
# make class predictions with the model
predictions = model.predict(x)
for i in range(785, 800):
	print(f'{x[i].tolist()} => {predictions[i]} expected {y[i]}')