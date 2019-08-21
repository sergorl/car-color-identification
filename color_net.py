from keras.models import Model, load_model
from keras.optimizers import SGD
from keras.layers import BatchNormalization, Lambda, Input, Dense, \
                         Convolution2D, MaxPooling2D, \
                         Dropout, Flatten
from keras.layers.merge import Concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import cv2
import numpy as np
import keras
import tensorflow as tf


class CarColorNet:

    def __init__(self, numClasses=6, imageWidth=256, imageHeight=256):

        self.classes = {}
        self.numClasses = numClasses
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight

        input_image = Input(shape=(self.imageWidth, self.imageHeight, 3))

        # ------------------------------------ TOP BRANCH ------------------------------------
        # first top convolution layer
        top_conv1 = Convolution2D(filters=48, kernel_size=(11, 11), strides=(4, 4),
                                  input_shape=(self.imageWidth, self.imageHeight, 3), activation='relu')(input_image)
        top_conv1 = BatchNormalization()(top_conv1)
        top_conv1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(top_conv1)

        # second top convolution layer
        # split feature map by half
        top_top_conv2 = Lambda(lambda x: x[:, :, :, :24])(top_conv1)
        top_bot_conv2 = Lambda(lambda x: x[:, :, :, 24:])(top_conv1)

        top_top_conv2 = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                                      padding='same')(top_top_conv2)
        top_top_conv2 = BatchNormalization()(top_top_conv2)
        top_top_conv2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(top_top_conv2)

        top_bot_conv2 = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                                      padding='same')(top_bot_conv2)
        top_bot_conv2 = BatchNormalization()(top_bot_conv2)
        top_bot_conv2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(top_bot_conv2)

        # third top convolution layer
        # concat 2 feature map
        top_conv3 = Concatenate()([top_top_conv2, top_bot_conv2])
        top_conv3 = Convolution2D(filters=192, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                                  padding='same')(top_conv3)

        # fourth top convolution layer
        # split feature map by half
        top_top_conv4 = Lambda(lambda x: x[:, :, :, :96])(top_conv3)
        top_bot_conv4 = Lambda(lambda x: x[:, :, :, 96:])(top_conv3)

        top_top_conv4 = Convolution2D(filters=96, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                                      padding='same')(top_top_conv4)
        top_bot_conv4 = Convolution2D(filters=96, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                                      padding='same')(top_bot_conv4)

        # fifth top convolution layer
        top_top_conv5 = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                                      padding='same')(top_top_conv4)
        top_top_conv5 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(top_top_conv5)

        top_bot_conv5 = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                                      padding='same')(top_bot_conv4)
        top_bot_conv5 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(top_bot_conv5)

        # ------------------------------------ TOP BOTTOM ------------------------------------
        # first bottom convolution layer
        bottom_conv1 = Convolution2D(filters=48, kernel_size=(11, 11), strides=(4, 4),
                                     input_shape=(224, 224, 3), activation='relu')(input_image)
        bottom_conv1 = BatchNormalization()(bottom_conv1)
        bottom_conv1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(bottom_conv1)

        # second bottom convolution layer
        # split feature map by half
        bottom_top_conv2 = Lambda(lambda x: x[:, :, :, :24])(bottom_conv1)
        bottom_bot_conv2 = Lambda(lambda x: x[:, :, :, 24:])(bottom_conv1)

        bottom_top_conv2 = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                                         padding='same')(bottom_top_conv2)
        bottom_top_conv2 = BatchNormalization()(bottom_top_conv2)
        bottom_top_conv2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(bottom_top_conv2)

        bottom_bot_conv2 = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                                         padding='same')(bottom_bot_conv2)
        bottom_bot_conv2 = BatchNormalization()(bottom_bot_conv2)
        bottom_bot_conv2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(bottom_bot_conv2)

        # third bottom convolution layer
        # concat 2 feature map
        bottom_conv3 = Concatenate()([bottom_top_conv2, bottom_bot_conv2])
        bottom_conv3 = Convolution2D(filters=192, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                                     padding='same')(bottom_conv3)

        # fourth bottom convolution layer
        # split feature map by half
        bottom_top_conv4 = Lambda(lambda x: x[:, :, :, :96])(bottom_conv3)
        bottom_bot_conv4 = Lambda(lambda x: x[:, :, :, 96:])(bottom_conv3)

        bottom_top_conv4 = Convolution2D(filters=96, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                                         padding='same')(bottom_top_conv4)
        bottom_bot_conv4 = Convolution2D(filters=96, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                                         padding='same')(bottom_bot_conv4)

        # fifth bottom convolution layer
        bottom_top_conv5 = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                                         padding='same')(bottom_top_conv4)
        bottom_top_conv5 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(bottom_top_conv5)

        bottom_bot_conv5 = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                                         padding='same')(bottom_bot_conv4)
        bottom_bot_conv5 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(bottom_bot_conv5)

        # ---------------------------------- CONCATENATE TOP AND BOTTOM BRANCH ------------------------------------
        conv_output = Concatenate()([top_top_conv5, top_bot_conv5, bottom_top_conv5, bottom_bot_conv5])

        # Flatten
        flatten = Flatten()(conv_output)

        # Fully-connected layer
        FC_1 = Dense(units=4096, activation='relu')(flatten)
        FC_1 = Dropout(0.6)(FC_1)
        FC_2 = Dense(units=4096, activation='relu')(FC_1)
        FC_2 = Dropout(0.6)(FC_2)
        output = Dense(units=self.numClasses, activation='softmax')(FC_2)

        self.model = Model(inputs=input_image, outputs=output)
        sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self,
              pathToTrainSet,
              pathToValidSet,
              pathToSaveModel,
              pathToSaveWeights,
              pathToSaveClassIndexes,
              epochs=10,
              batchSize=32,
              stepsPerEpoch=200,
              validationSteps=1000):

        keras.backend.get_session().run(tf.global_variables_initializer())

        checkpoint = ModelCheckpoint(pathToSaveWeights,
                                     monitor='val_acc', verbose=1,
                                     save_best_only=True, mode='max')

        trainDataGen = ImageDataGenerator(rescale=1.0/255, shear_range=0.2,
                                          zoom_range=0.3, horizontal_flip=True)

        validDataGen = ImageDataGenerator(rescale=1.0/255)

        trainSet = trainDataGen.flow_from_directory(
                pathToTrainSet,
                target_size=(self.imageWidth, self.imageHeight),
                batch_size=batchSize,
                class_mode='categorical'
        )

        self.classes = {v: k for k, v in trainSet.class_indices.items()}
        np.save(pathToSaveClassIndexes, self.classes, allow_pickle=True)

        validSet = validDataGen.flow_from_directory(
                pathToValidSet,
                target_size=(self.imageWidth, self.imageHeight),
                batch_size=batchSize,
                class_mode='categorical'
        )

        self.model.fit_generator(
            trainSet,
            steps_per_epoch=stepsPerEpoch,
            epochs=epochs,
            validation_data=validSet,
            validation_steps=validationSteps//batchSize,
            callbacks=[checkpoint])

        print('============================ Saving is here ============================')
        self.model.save(pathToSaveModel)

    @staticmethod
    def load(pathToModel, pathToClassIndexes):

        model = load_model(pathToModel)

        layers = model.layers
        inputShape, outputShape = layers[0].input_shape, layers[-1].output_shape,

        imageWidth, imageHeight = inputShape[1], inputShape[2]
        numClasses = outputShape[1]

        net = CarColorNet(numClasses, imageWidth, imageHeight)
        net.classes = np.load(pathToClassIndexes).item()

        return net

    def predictOneImage(self, image):

        if isinstance(image, str):
            frame = cv2.imread(image)
        else:
            frame = image

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.imageWidth, self.imageHeight))

        frame = np.expand_dims(frame, axis=0)

        # cv2.imshow("boxed", frame[0, :, :, :])
        # cv2.waitKey(0)

        frame = np.asarray(frame, dtype='float32')/255

        probs = self.model.predict(frame)
        ind = probs.argmax(axis=-1)[0]

        return self.classes[ind]


if __name__ == '__main__':

    # Train net
    net = CarColorNet(numClasses=6)
    net.train(pathToTrainSet='/home/sergorl/cars/train',
              pathToValidSet= '/home/sergorl/cars/valid',
              pathToSaveModel='/home/sergorl/cars/car_color_net.h5',
              pathToSaveWeights='/home/sergorl/cars/color_weights.hdf5',
              pathToSaveClassIndexes='/home/sergorl/cars/class_index.npy')
