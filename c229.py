from numpy import loadtxt 
from keras.models import Sequential
from keras.layers import Dense 
# load the dataset 
dataset = loadtxt('diabetes_dataset.csv', delimiter=',') 
# split into input (x) and output (y) 

x = dataset[:,0:8] 
y = dataset[:,8] 
# define the keras model 
model = Sequential() 
model.add(Dense(12, input_dim=8, activation='relu')) 
model.add(Dense(8, activation='relu')) 
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentopy', metrics=['accuracy'])
model.fit(x,y, epochs=500, batch_size=100)

predictions = model.predict_classes(x)
for i in range(5):
	print(f'{x[i].tolist()} => {prediction[1]} expected {y[i]}')