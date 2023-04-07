import os
import cv2
import glob
import numpy as np
from sklearn.model_selection import KFold
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import Sequential


def load_preprocess(base_directory): # A function "base_directory" is created
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'] # A list of emotions are defined
    images = [] # An empty list for images is initialized to store the image data
    expressions = [] # An empty list for expressions is initialized to store the expressions
    
    # This function loops through the emotions from the emotions list
    for i, emotion in enumerate(emotions):
        category_path = os.path.join(base_directory, emotion, '*.jpg')
        for img_path in glob.glob(category_path):  #This looks for images through the directory for each emotion and returns the list of images that match the emotion specified in the "category_path" variable
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # For the image found, "cv2" reads the image data and converts it into grayscale
            img = cv2.resize(img, (48, 48)) # This resizes the image to (48,48) pixels
            img = img[..., np.newaxis] / 255.0 
            images.append(img) # The preprocessed image is appended into "images"- list
            expressions.append(i) # The corresponding emotion label is appended into "expressions" list

    images = np.array(images, dtype='float32') # The "images" list is converted into numpy arrays of dtype "float32"
    expressions = to_categorical(expressions) # The expressions array is encoded
    return images, expressions # This returns the preprocessed image data, and the corresponding encoded labels as numpy arrays

def build_model(): # "build_model" is defined
    model = Sequential() # A sequential object is created
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1))) # A "Conv2D" layer is added with 32 filters, kernel size of (3,3), an activation function, and input shape to map the outputs of feature map
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu')) # A "Conv2D" layer is added with 32 filters, kernel size of (3,3), an activation function, to map the outputs of previous layer
    model.add(MaxPooling2D(pool_size=(2, 2))) # "MaxPooling2D" is added to reduce the size of spatial dimensions by giving the pool size of (2,2)
    model.add(Dropout(0.25)) # "Dropout" of 0.25 is added to reduce overfitting by dropping 25% of connections between layers
    model.add(Flatten()) # "Flatten" function is added to flatten the feature map into 1D vector
    model.add(Dense(128, activation='relu')) # "Dense" function is added to apply linear transformation and outputs new vetor of size 128
    model.add(Dropout(0.5)) # "droupout" of 50% is added
    model.add(Dense(7, activation='softmax')) # A dense layer with 7 units is added, representing the probability pf distribution over 7 expressions
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy']) # The model is compiled over cross-entropy loss, Adam optimizer with learning rate of "0.0001", and accuracy metrics
    return model # The model is returned

def train_and_evaluate_model(images, labels, train_index, test_index, fold_number): # The function "train_and_evaluate_model" is defined with 5 parameters, "images", "labels", "train_index", "test_index", and "fold_number"
    model = build_model() # The "build_model" function s=is called to create an new CNN model
    train_images, train_labels = images[train_index], labels[train_index] # The "train_images", and "train_labels" lists are created to store the images and corresponding labels from "train_index" array
    test_images, test_labels = images[test_index], labels[test_index] # The "test_images", and "test_labels" lists are created to store the images and corresponding labels from "test_index" array
    model.fit(train_images, train_labels, batch_size=32, epochs=30, verbose=1, validation_data=(test_images, test_labels)) # The model is trained using  "fit"  with "train_images", and "train_labels" as inputs, size of 32, 30 epoches and verbrose of 1
    results = model.evaluate(test_images, test_labels, verbose=0) # "evaluate" method is used to evaluate the data
    print(f"Fold {fold_number} Results - Loss: {results[0]} - Accuracy: {results[1]}") # the results are printed by showing fold numbers, loss, and accuracy of model
    return results[1] # returns "results" 


def main(): # main function is defined
    train_directory = 'images/train' # The "train_directory" is set to paths of training directories containing image data
    val_directory='images/validation' # The "val_directory" is set to paths of validation directories containing image data
    images, labels = load_preprocess(train_directory) # The "load_preprocessor" function is called to preprocess the image data and labels from the "train_directory"
    x_val,y_val = load_preprocess(val_directory) # The "load_preprocessor" function is called to preprocess the image data and labels from the "val_directory"
    kfold = KFold(n_splits=5, shuffle=True) #The "kfold" function is called to create k-folds with 5 folds and with shuffles
    accuracies = [] # The "accuracies" list is created to store the accuracies of each fold
    fold_number = 1 # "fold_number" is created to keep track of folds and is set to 1
    # A for loop is used to iterate over each fold of K-fold cross-validator
    for train_index, test_index in kfold.split(images):
        accuracy = train_and_evaluate_model(images, labels, train_index, test_index, fold_number) # On each iteratin "train_and_evaluate_model" function is called with current folds training and validation indices
        accuracies.append(accuracy) # The accuracy value returned is appended to "accuracy" list
        fold_number += 1 # The fold number is incremented
    mean_accuracy = np.mean(accuracies)- # The mean accuracy across all folds is calculated
    print(f"Mean accuracy: {mean_accuracy}") # The mean accuracy is printed

# This function ensures that the main function only executes when the scrip runs directly
if __name__ == '__main__':
    main()
#references: OpenCV Library, OpenCV Library, 2015. http://opencv.org.
#"Walt, S., Colbert, S. C., & Varoquaux, G. (2011). The NumPy Array: A Structure for Efficient Numerical Computation. Computing in Science & Engineering, 13(2), 22–30. https://doi.org/10.1109/mcse.2011.37"
#"Pedregosa, F. et al. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research 12, 2825–2830 (2011)."
#Chollet, F. et al. Keras. https://keras.io (2015)."
#Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization. arXiv preprint arXiv:1412.6980
#Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer, New York, NY, USA.
