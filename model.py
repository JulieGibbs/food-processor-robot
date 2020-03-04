from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#neural network which will be added to
classifier = Sequential()

#adding conv layer + pooling layer
classifier.add(Convolution2D(32, (3,1), input_shape=(200, 200, 1), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

#2nd conv + pooling layer
classifier.add(Convolution2D(32, (3, 3), activation="relu")) #input shape is transferred from previous
classifier.add(MaxPooling2D(pool_size=(2, 2)))

#flattening layers
classifier.add(Flatten())

#fully connected layer
classifier.add(Dense(units=128, activation="relu"))
classifier.add(Dense(units=2, activation="softmax")) #2 outputs

#compilation
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# keras.io/preprocessing/image/
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = test_datagen.flow_from_directory('./train',
                                                target_size=(200, 200),
                                                batch_size=9,
                                                color_mode='grayscale',
                                                class_mode='categorical')

test_set = test_datagen.flow_from_directory('./test',
                                                target_size=(200, 200),
                                                batch_size=9,
                                                color_mode='grayscale',
                                                class_mode='categorical')

#training the model
classifier.fit_generator(
    training_set,
    steps_per_epoch=473, #images in training set
    epochs=3,
    validation_data=test_set,
    validation_steps= 18 #images in test set
)

#saving model
model_json = classifier.to_json()
with open("model-bw.json", "w") as json_file:
    json_file.write(model_json)
classifier.save_weights('model-bw.h5')



