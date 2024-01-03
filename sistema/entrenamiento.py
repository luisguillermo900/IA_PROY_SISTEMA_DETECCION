import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Definir la arquitectura del modelo CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Configurar el generador de datos
datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, validation_split=0.2)

# Cargar y dividir los datos en conjuntos de entrenamiento y validación
train_generator = datagen.flow_from_directory('data/parpadeos', target_size=(64, 64), batch_size=32, class_mode='binary', subset='training')
validation_generator = datagen.flow_from_directory('data/parpadeos', target_size=(64, 64), batch_size=32, class_mode='binary', subset='validation')

# Entrenar el modelo
model.fit(train_generator, epochs=25, validation_data=validation_generator)

# Guardar el modelo entrenado
model.save('modelo_parpadeos.h5')
