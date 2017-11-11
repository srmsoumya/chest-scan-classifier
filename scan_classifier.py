import argparse
import os
import random
import sys
from collections import Counter
import numpy as np
import pandas as pd
from PIL import Image
from pprint import pprint as pp

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.optimizers import SGD, Adam
from keras import regularizers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler


parser = argparse.ArgumentParser(description='Classify chest scan images of different patients.')
parser.add_argument('--train', action='store_true', help='Train the model', default=False)
parser.add_argument('--predict', action='store_true', help='Predict a particular class', default=False)
parser.add_argument('--evaluate', action='store_true', help='Evaluate the model', default=False)
parser.add_argument('--model', help='Model Path')
parser.add_argument('--cls', help='Image class')
args = parser.parse_args()


class ChestScanClassifier(object):
    ''' Initialise the parameters for the model '''
    def __init__(self,
                 img_size=(299, 299),
                 num_epochs=30,
                 batch_size=32,
                 num_fixed_layers=50,
                 train_dir='data/train',
                 validation_dir='data/validation',
                 test_dir='data/test'):
        self.img_size = img_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_fixed_layers = num_fixed_layers

        self.train_dir = train_dir
        self.validation_dir = validation_dir
        self.test_dir = test_dir
        self.num_classes = len(os.listdir(self.train_dir))
        self.train_size = self.data_stats(self.train_dir)
        self.validation_size = self.data_stats(self.validation_dir)
        self.test_size = self.data_stats(self.test_dir)

        print('\nImage Size: ({}, {})'.format(self.img_size[0], self.img_size[1]))
        print('Number of Epochs: {}'.format(self.num_epochs))
        print('Number of classes: {}'.format(self.num_classes))
        print('Training Data Size: {}'.format(self.train_size))
        print('Validation Data Size: {}'.format(self.validation_size))
        print('Test Data Size: {}\n'.format(self.test_size))

    def create_base_model(self):
        model = InceptionV3(weights='imagenet', include_top=False)
        for layer in model.layers:
            layer.trainable = False
        return model

    def add_custom_layers(self, base_model):
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.005))(x)
        x = Dense(1024, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.005))(x)
        y = Dense(self.num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=y)
        return model

    def rebase_base_model(self, model):
        for layer in model.layers[:self.num_fixed_layers]:
            layer.trainable = False
        for layer in model.layers[self.num_fixed_layers:]:
            layer.trainable = True
        return model

    def create_class_weights(self, y, smooth_factor=0.15):
        counter = Counter(y)
        if smooth_factor > 0:
            p = max(counter.values()) * smooth_factor
            for k in counter.keys():
                counter[k] += p
        majority = max(counter.values())
        return {cls: float(majority / count) for cls, count in counter.items()}

    def create_data_generator(self,
                              preprocessing_function=preprocess_input,
                              rotation_range=4,
                              width_shift_range=0.1,
                              height_shift_range=0.1,
                              shear_range=0.05,
                              zoom_range=0.1,
                              fill_mode='nearest',
                              horizontal_flip=False,
                              vertical_flip=False):

        train_image_gen = ImageDataGenerator(
            preprocessing_function=preprocessing_function,
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            fill_mode=fill_mode,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip
        )
        test_image_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

        return train_image_gen, test_image_gen

    def data_stats(self, directory):
        counter = 0
        for d in os.listdir(directory):
            counter += len(os.listdir(os.path.join(directory, d)))
        return counter

    def trainer(self, train_gen, validation_gen, call):
        if call == 'fp':
            print('\nCreating the Model for first Pass...')
            base_model = self.create_base_model()
            model = self.add_custom_layers(base_model)
            model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
            path = 'model/chest_scan_classifier_fp.h5'
        else:
            print('\nLoading the Model for second pass...')
            model = load_model('model/chest_scan_classifier_fp.h5')
            model = self.rebase_base_model(model)
            adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
            model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
            path = 'model/chest_scan_classifier_sp.h5'

        # if not os.path.exists('model'):
        #     os.mkdir('model')
        weights = self.create_class_weights(train_gen.classes, 0.15)
        pp(weights)

        print('\nSetting up a check point...')
        cp = ModelCheckpoint(path,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=False,
                             mode='auto',
                             period=1)

        print('\nTraining the model...')
        model.fit_generator(
            train_gen,
            epochs=self.num_epochs,
            steps_per_epoch=(self.train_size // self.batch_size),
            validation_data=validation_gen,
            validation_steps=(self.validation_size // self.batch_size) + 1,
            class_weight=weights,
            callbacks=[cp]
        )

        test_image_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
        print('\nTesting the model')
        results = model.evaluate_generator(
            test_image_gen.flow_from_directory(self.test_dir, target_size=self.img_size, batch_size=self.batch_size, shuffle=False),
            steps=(self.test_size // self.batch_size) + 1
        )
        print('Model: Loss => {}, Accuracy => {}'.format(results[0], results[1]))

    def train(self):
        train_image_gen, test_image_gen = self.create_data_generator()
        train_gen = train_image_gen.flow_from_directory(self.train_dir, target_size=self.img_size, batch_size=self.batch_size)
        validation_gen = test_image_gen.flow_from_directory(self.validation_dir, target_size=self.img_size, batch_size=self.batch_size)

        print('\nRunning First Pass...')
        self.trainer(train_gen, validation_gen, 'fp')
        print('\nRunning the Second Pass...')
        self.trainer(train_gen, validation_gen, 'sp')

    def predict(self, img, model):
      if img.size != self.img_size:
        img = img.resize(self.img_size)

      x = img_to_array(img)
      x = np.expand_dims(x, axis=0)
      x = preprocess_input(x)
      y = model.predict(x)
      return y[0]

    def image_predict(self, img, model):
        model = load_model(model)
        # Get all the class labels
        classes = []
        for c in sorted(os.listdir('data/test/')):
            classes.append(c)
        img = Image.open(img)
        prediction = self.predict(img, model)
        res = sorted(zip(classes, prediction), key=lambda x: x[1], reverse=True)[:5]
        print(file, res)

    def class_predict(self, cls, model):
        model = load_model(model)

        # Get all the class labels
        classes = []
        for c in sorted(os.listdir('data/test/')):
            classes.append(c)
        path = os.path.join('data/test', cls)

        for file in os.listdir(path):
            img = Image.open(os.path.join(path, file))
            prediction = self.predict(img, model)
            res = sorted(zip(classes, prediction), key=lambda x: x[1], reverse=True)[:2]
            print(file, res)

    def matrix(self, model):
        model = load_model(model)
        # Get all the class labels
        classes = []
        for c in sorted(os.listdir('data/test/')):
            classes.append(c)

        eval_matrix_train = {}
        eval_matrix_validation = {}
        eval_matrix_test = {}

        print('\nEvaluating Training Set')
        for d in os.listdir('data/train'):
            eval_matrix_train[d] = {
                'top_1': 0,
                'top_5': 0,
                'total': 0
            }
            for file in os.listdir(os.path.join('data/train', d)):
                img = Image.open(os.path.join('data/train', d, file))
                prediction = self.predict(img, model)
                result = sorted(zip(classes, prediction), key=lambda x: x[1], reverse=True)[:5]
                if d == result[0][0]:
                    eval_matrix_train[d]['top_1'] += 1
                if d in [k for k, v in result]:
                    eval_matrix_train[d]['top_5'] += 1
                eval_matrix_train[d]['total'] += 1
        train_df = pd.DataFrame(eval_matrix_train).transpose()
        train_df['top_1_%'] = (train_df['top_1'] / train_df['total']) * 100
        train_df['top_5_%'] = (train_df['top_5'] / train_df['total']) * 100
        print(train_df.round(2))

        print('\nEvaluating Validation Set')
        for d in os.listdir('data/validation'):
            eval_matrix_validation[d] = {
                'top_1': 0,
                'top_5': 0,
                'total': 0
            }
            for file in os.listdir(os.path.join('data/validation', d)):
                img = Image.open(os.path.join('data/validation', d, file))
                prediction = self.predict(img, model)
                result = sorted(zip(classes, prediction), key=lambda x: x[1], reverse=True)[:5]
                if d == result[0][0]:
                    eval_matrix_validation[d]['top_1'] += 1
                if d in [k for k, v in result]:
                    eval_matrix_validation[d]['top_5'] += 1
                eval_matrix_validation[d]['total'] += 1
        validation_df = pd.DataFrame(eval_matrix_validation).transpose()
        validation_df['top_1_%'] = (validation_df['top_1'] / validation_df['total']) * 100
        validation_df['top_5_%'] = (validation_df['top_5'] / validation_df['total']) * 100
        print(validation_df.round(2))



        print('\nEvaluating Test Set')
        for d in os.listdir('data/test'):
            eval_matrix_test[d] = {
                'top_1': 0,
                'top_5': 0,
                'total': 0
            }
            for file in os.listdir(os.path.join('data/test', d)):
                img = Image.open(os.path.join('data/test', d, file))
                prediction = self.predict(img, model)
                result = sorted(zip(classes, prediction), key=lambda x: x[1], reverse=True)[:5]
                if d == result[0][0]:
                    eval_matrix_test[d]['top_1'] += 1
                if d in [k for k, v in result]:
                    eval_matrix_test[d]['top_5'] += 1
                eval_matrix_test[d]['total'] += 1

        test_df = pd.DataFrame(eval_matrix_test).transpose()
        test_df['top_1_%'] = (test_df['top_1'] / test_df['total']) * 100
        test_df['top_5_%'] = (test_df['top_5'] / test_df['total']) * 100
        print(test_df.round(2))

        writer = pd.ExcelWriter('/output/evaluation-matrix.xlsx')
        train_df.to_excel(writer, 'Train')
        validation_df.to_excel(writer, 'Validation')
        test_df.to_excel(writer, 'Test')
        writer.save()

if __name__ == '__main__':
    csclf = ChestScanClassifier()
    if args.train:
        print('\n Starting Training Phase\n')
        csclf.train()
    elif args.predict:
        print('\nStarting Prediction Phase\n')
        csclf.class_predict(args.cls, args.model)
    elif args.evaluate:
        species.matrix(args.model)
    else:
        print('Enter a valid argument parser')

