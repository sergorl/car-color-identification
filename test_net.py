import os
import cv2
import numpy as np
from color_net import CarColorNet

colors = {'белый': 'white',
          'черный': 'black',
          'красный': 'red',
          'синий': 'blue',
          'оранжевый': 'orange',
          'коричневый': 'brown',
          'серый': 'grey',
          'темно-серый': 'dark grey',
          'зеленый': 'green'
          }


def testNetAndCreateVideo(net,
                          pathDirWithOriginalImages,
                          pathToImageBoxes,
                          pathToSaveVideo):

    images = np.load(pathToImageBoxes).item()
    n = float(len(images))

    out = cv2.VideoWriter(pathToSaveVideo, cv2.VideoWriter_fourcc(*"MJPG"), 1, (1920, 1080))

    for i, imageName in enumerate(images):

        pathToImage = os.path.join(pathDirWithOriginalImages, imageName)
        sourceImage = cv2.imread(pathToImage)

        if sourceImage is not None:

            boxes = images[imageName]

            for box in boxes:
                xtl, ytl, xbr, ybr = box['xtl'], box['ytl'], box['xbr'], box['ybr']
                trueLabel = box['label']

                subImage = sourceImage[ytl:ybr, xtl:xbr]

                # predictLabel = trueLabel
                predictLabel = net.predictOneImage(subImage)

                colorLabel = (0, 255, 0) if predictLabel == trueLabel else (0, 0, 255)

                cv2.rectangle(sourceImage, (xtl, ytl), (xbr, ybr), colorLabel, 2)

                cv2.putText(sourceImage, colors[predictLabel],
                            (xtl, ytl),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, colorLabel, 2)

            # cv2.imshow("boxed", sourceImage)
            # cv2.waitKey(0)

            out.write(sourceImage)

            print('Processing {0:.2f}%'.format(float(i)/n * 100))

    out.release()

    print('Video is saved into {}'.format(pathToSaveVideo))


def testNetOnImages(net, pathToDirWithDatasetsOfImages):

    count, countTrueLabels = 0, 0

    for dirpath, _dirnames, filenames in os.walk(pathToDirWithDatasetsOfImages):
        trueLabel = dirpath.split('/')[-1]

        for file in filenames:

            predictLabel = net.predictOneImage(os.path.join(dirpath, file))

            print('{}: true = {}, predict = {} is {}'
                  .format(file, trueLabel, predictLabel,
                          'OK.' if predictLabel == trueLabel else 'Wrong!'))

            if predictLabel == trueLabel:
                countTrueLabels += 1

            count += 1

    print('Rate is {0:.2f}'.format(float(countTrueLabels) / float(count) * 100))


if __name__ == '__main__':

    # Create net and load pre learned weights
    net = CarColorNet(numClasses=6)
    net.classes = np.load('/home/sergorl/cars/class_index.npy').item()
    net.loadWeights('/home/sergorl/cars/color_weights.hdf5')

    # Train net
    # net.train(pathToTrainSet='/home/sergorl/cars/train',
    #           pathToValidSet='/home/sergorl/cars/valid',
    #           pathToSaveModel='/home/sergorl/cars/car_color_net.h5',
    #           pathToSaveWeights='/home/sergorl/cars/color_weights.hdf5',
    #           pathToSaveClassIndexes='/home/sergorl/cars/class_index.npy')

    # Manual validation on images from pathToValidSet
    # testNetOnImages(net, pathToDirWithDatasetsOfImages='/home/sergorl/cars/valid')

    # Manual validation on images with making video from original images
    testNetAndCreateVideo(net,
                          pathDirWithOriginalImages='/home/sergorl/cars/dataset1/images',
                          pathToImageBoxes='/home/sergorl/cars/car_boxes_1.npy',
                          pathToSaveVideo='/home/sergorl/cars/cars1.mp4')

    testNetAndCreateVideo(net,
                          pathDirWithOriginalImages='/home/sergorl/cars/dataset2/images',
                          pathToImageBoxes='/home/sergorl/cars/car_boxes_2.npy',
                          pathToSaveVideo='/home/sergorl/cars/cars2.mp4')
