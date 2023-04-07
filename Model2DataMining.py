import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping



train_dir = 'C:/Users/user/Desktop/Data Mining/images/train'
val_dir = 'C:/Users/user/Desktop/Data Mining/images/validation'
batch_size = 64
target_size = (48, 48)

# Load and preprocess the dataset
# The variable "train_datagen" uses "ImageDataGenerator" from keras API to define data augmentation and normalization for image data
train_datagen = ImageDataGenerator(rescale=1./255, # The "rescale" parameter is set to 1/255 to normalize pixels between 0 and 1
                                   rotation_range=20, # All the parameters specify transformations needed to be applied to the input  image during training
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.1,
                                   zoom_range=0.1,
                                   horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255) # The "val_datagen" is used to valiate data and only applies normalization to the input images

# The "train_generator" reads images form "train_dir" directory
train_generator = train_datagen.flow_from_directory(train_dir, # The "train_generator" preprocesses data according to "train_datagen"
                                                    target_size=target_size, # The "target_size" specifies the desired size of the output images
                                                    batch_size=batch_size, # The "batch_size" specifies the number of images to `be included in each batch
                                                    color_mode='grayscale', # The "color_mode" converts images into gray scale
                                                    class_mode='categorical') # "class_mode" specifies data should be encoded

# The "val_generator" reads images form "val_dir" directory
val_generator = val_datagen.flow_from_directory(val_dir, # The "val_generator" preprocesses data according to "val_datagen"
                                                target_size=target_size, # The "target_size" specifies the desired size of the output images
                                                batch_size=batch_size, # The "batch_size" specifies the number of images to `be included in each batch
                                                color_mode='grayscale', # The "color_mode" converts images into gray scale
                                                class_mode='categorical') # "class_mode" specifies data should be encoded

# Define the model architecture
model = Sequential([ # Initializing a sequential model
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)), # A 2D convolation layer is added with 32 layers, size (3,3), an activation function and input of grayscale image
    MaxPooling2D((2, 2)), # Max pooling of (2,2) is added 
    Conv2D(64, (3, 3), activation='relu'), # Another 2D convolation layer ia added with 64 layers, size of (3,3), an activation function
    MaxPooling2D((2, 2)), # Another Max pooling of (2,2) is added 
    Conv2D(128, (3, 3), activation='relu'), # Another 2D convolation layer ia added with 128 layers, size of (3,3), an activation function
    MaxPooling2D((2, 2)), # Another Max pooling of (2,2) is added 
    Flatten(), # Falttening the output from previous layer into 1D vector
    Dense(256, activation='relu'), # Adds a fully connected layer with 256 units and an activation function.
    Dropout(0.5), # Applies dropout of 50%
    Dense(128, activation='relu'), # Adds another fully connected layer with 128 units and an activation function.
    Dropout(0.5), # Applies another dropout of 50%
    Dense(7, activation='softmax') # Adds output layer with 7 layers corresponding 7 facial expressions
])

model.summary() # Prints model summary

# Train the model
model.compile(loss='categorical_crossentropy', # Compiles the model by setting categorical cross-entropy loss, Adam optimizer and accuracy metrics
              optimizer='adam', 
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10) # "early_stopping" is added to stop the training if the loss does not improve after 10 epochs

training = model.fit(train_generator, # Trains the model by taking "train_generator" as input
                    steps_per_epoch=len(train_generator), # Sets steps per each epoch according to the length of "train_generator"
                    epochs=50, # Sets epochs to 50
                    validation_data=val_generator, # Trains the model by taking "val_generator" as input
                    validation_steps=len(val_generator), # Sets validation steps according to the length of "val_generator"
                    callbacks=[early_stopping])

# Evaluate the model
score = model.evaluate(val_generator, steps=len(val_generator)) # Evaluating the trained model on validation dataset
print('Validation loss:', score[0]) # Prints the validation loss 
print('Validation accuracy:', score[1]) # Prints the validation accuracy

#plot the accuracy and loss curves
plt.plot(training.history['accuracy']) # Plots training accuracy for each epoch
plt.plot(training.history['val_accuracy']) # Plots validation accuracy for each epoch
plt.title('Model accuracy') # Sets the plot title as "Model Accuracy"
plt.ylabel('Accuracy') # Sets label for y-axis as "Acuuracy"
plt.xlabel('Epoch') # Sets label for x-axis as "Epoch"
plt.legend(['Train', 'Val'], loc='upper left') # Adding leged with labels as "Train" and "Val" and sets it at upper left 
plt.show() # Shows the plot

plt.plot(training.history['loss']) # Plots training loss for each epoch
plt.plot(training.history['val_loss']) # Plots validation loss for each epoch
plt.title('Model loss') # Sets the plot title as "Model loss"
plt.ylabel('Loss') # Sets label for y-axis as "Loss"
plt.xlabel('Epoch') # Sets label for x-axis as "Epoch"
plt.legend(['Train', 'Val'], loc='upper left') # Adding leged with labels as "Train" and "Val" and sets it at upper left 
plt.show() # Shows the plot
