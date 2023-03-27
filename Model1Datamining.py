import os
import cv2
import glob
import numpy as np
from sklearn.model_selection import KFold
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import Sequential


def load_preprocess(base_directory):
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    images = []
    expressions = []

    for i, emotion in enumerate(emotions):
        category_path = os.path.join(base_directory, emotion, '*.jpg')
        for img_path in glob.glob(category_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (48, 48))
            img = img[..., np.newaxis] / 255.0
            images.append(img)
            expressions.append(i)

    images = np.array(images, dtype='float32')
    expressions = to_categorical(expressions)
    return images, expressions

def build_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
    return model

def train_and_evaluate_model(images, labels, train_index, test_index, fold_number):
    model = build_model()
    train_images, train_labels = images[train_index], labels[train_index]
    test_images, test_labels = images[test_index], labels[test_index]
    model.fit(train_images, train_labels, batch_size=32, epochs=30, verbose=1, validation_data=(test_images, test_labels))
    results = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Fold {fold_number} Results - Loss: {results[0]} - Accuracy: {results[1]}")
    return results[1]

def main():
    train_directory = 'images/train'
    val_directory='images/validation'
    images, labels = load_preprocess(train_directory)
    x_val,y_val = load_preprocess(val_directory)
    kfold = KFold(n_splits=5, shuffle=True)
    accuracies = []
    fold_number = 1
    for train_index, test_index in kfold.split(images):
        accuracy = train_and_evaluate_model(images, labels, train_index, test_index, fold_number)
        accuracies.append(accuracy)
        fold_number += 1
    mean_accuracy = np.mean(accuracies)
    print(f"Mean accuracy: {mean_accuracy}")
    
if __name__ == '__main__':
    main()
