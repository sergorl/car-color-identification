import random
import os
import shutil
from xml.dom import minidom
import numpy as np
import cv2


class DatasetBuilder:

    def __init__(self,
                 pathToXMLAnnotation,
                 pathToDirWithOriginalImages,
                 pathToDataSetDir,
                 pathToSaveImageBoxes,
                 strategyDivideSet='everyClassLabel',
                 partTrain=0.7):

        self.exceptLabels = set(['человек', 'сотрудник'])
        self.strategyDivideSet = strategyDivideSet

        self.partTrain = partTrain

        self.pathToXMLAnnotation = pathToXMLAnnotation
        self.pathToDirWithOriginalImages = pathToDirWithOriginalImages
        self.pathToSaveImageBoxes = pathToSaveImageBoxes

        self.pathToTrainSet = os.path.join(pathToDataSetDir, 'train')
        self.pathToValidSet = os.path.join(pathToDataSetDir, 'valid')

        self.xmlDoc = minidom.parse(pathToXMLAnnotation)
        self.supportedClasses = self.selectLabels()

        images, self.labels = self.findImagesandLabels()

        self.countImages = 0

        datasets = self.divideOnTrainAndValidSet(images, partTrain, strategyDivideSet)

        self.createDirs(self.pathToTrainSet, self.labels)
        self.createDirs(self.pathToValidSet, self.labels)

        for trainSet, validSet in datasets:
            self.createDataset(pathToDirWithOriginalImages, trainSet, self.pathToTrainSet)
            self.createDataset(pathToDirWithOriginalImages, validSet, self.pathToValidSet)

        self.missedLabels = list(self.supportedClasses - self.labels)

        print('New data set is created from {}'.format(pathToDirWithOriginalImages))
        print('These classes are absent in the XML boxes: {} from {}'
              .format(self.missedLabels, self.pathToXMLAnnotation))

    def selectLabels(self):

        labels = self.xmlDoc.getElementsByTagName('label')
        names = set()

        for label in labels:
            name = label.getElementsByTagName('name')[0].firstChild.data
            if name not in self.exceptLabels:
                names.add(name)

        return names

    def findImagesandLabels(self):

        images, labels = [], set()
        collectOfImagesForSave = {}

        for tag in self.xmlDoc.getElementsByTagName('image'):
            img = tag.attributes

            # widthImage = float(img['width'].firstChild.nodeValue)
            # heightImage = float(img['height'].firstChild.nodeValue)

            imageName = img['name'].firstChild.nodeValue

            collectOfImagesForSave[imageName] = []

            boxes = tag.getElementsByTagName('box')

            for box in boxes:
                attr = {k: v for k, v in box.attributes.items()}
                if attr['label'] not in self.exceptLabels:
                    xtl, ytl = int(float(attr['xtl'])), int(float(attr['ytl']))
                    xbr, ybr = int(float(attr['xbr'])), int(float(attr['ybr']))

                    images.append(
                        {'imageName': imageName, 'xtl': xtl, 'ytl': ytl, 'xbr': xbr, 'ybr': ybr, 'label': attr['label']}
                    )

                    labels.add(attr['label'])

                    collectOfImagesForSave[imageName].append(
                        {'xtl': xtl, 'ytl': ytl, 'xbr': xbr, 'ybr': ybr, 'label': attr['label']}
                    )

        np.save(self.pathToSaveImageBoxes, collectOfImagesForSave, allow_pickle=True)

        return images, labels

    def groupImagesByClassLabel(self, images):

        groupOfImages = {label: [] for label in self.labels}

        for image in images:
            groupOfImages[image['label']].append(image)

        return groupOfImages

    def divideOnTrainAndValidSet(self, images, partTrain=0.7, strategyDivideSet='everyClassLabel'):
        if strategyDivideSet == 'everyClassLabel':
            return self.divideEveryClassLabelOnTrainAndValid(images, partTrain)
        elif strategyDivideSet == 'all':
            return self.divideAllOnTrainAndValid(images, partTrain)
        else:
            raise Exception('You should choose strategy of dividing ' \
                            'data set between \"everyClassLabel\" and ' \
                            'all')

    def divideEveryClassLabelOnTrainAndValid(self, images, partTrain=0.7):

        groupOfImages = self.groupImagesByClassLabel(images)

        datasets = []

        for _label, group in groupOfImages.items():
            dataset = self.divideAllOnTrainAndValid(group, partTrain)
            datasets.append(dataset[0])

        return datasets

    def divideAllOnTrainAndValid(self, images, partTrain=0.7):

        offset = int(partTrain * float(len(images)))

        random.shuffle(images)

        train = images[0:offset]
        valid = images[offset:]

        return [(train, valid)]

    def createDirs(self, pathToSaveDir, labels):

        if not os.path.exists(pathToSaveDir):
            os.makedirs(pathToSaveDir)

        for dirLabel in labels:
            pathToDirLabel = os.path.join(pathToSaveDir, dirLabel)
            if not os.path.exists(pathToDirLabel):
                os.makedirs(pathToDirLabel)

    def createDataset(self, pathToDirWithOriginalImages, images, pathToSaveDir):

        n = float(len(images))

        for i, image in enumerate(images):
            self.extractImageFromImageAndSaveIntoDir(pathToDirWithOriginalImages, image, pathToSaveDir)

            if self.strategyDivideSet == 'all':

                print('Processing image {0} with label {1}: {2:.2f} %'
                      .format(image['imageName'],
                        image['label'],
                        float(i+1)/n * 100)
                )

        if len(images) > 0 and self.strategyDivideSet == 'everyClassLabel':

            print('Class {} of images from {} is processed'
                  .format(images[0]['label'], self.pathToDirWithOriginalImages)
            )

    def extractImageFromImageAndSaveIntoDir(self, pathToDirWithOriginalImages, source, pathToSaveDir):

        sourceImage = cv2.imread(os.path.join(pathToDirWithOriginalImages, source['imageName']))

        xtl, ytl, xbr, ybr = source['xtl'], source['ytl'], source['xbr'], source['ybr']
        labelDir = source['label']

        subImage = sourceImage[ytl:ybr, xtl:xbr]

        imageName = os.path.splitext(source['imageName'])[0]

        cv2.imwrite(os.path.join(pathToSaveDir,
                                 labelDir,
                                 '{}_{}_{}.png'.format(imageName, source['label'], self.countImages)),
                                 subImage)

        self.countImages += 1


if __name__ == '__main__':

    # delete previous dirs of train and valid data sets
    shutil.rmtree('/home/sergorl/cars/train', ignore_errors=True)
    shutil.rmtree('/home/sergorl/cars/valid', ignore_errors=True)

    pathToXML1 = '/home/sergorl/cars/dataset1/annotations.xml'
    pathToXML2 = '/home/sergorl/cars/dataset2/annotations.xml'

    pathToImages1 = '/home/sergorl/cars/dataset1/images'
    pathToImages2 = '/home/sergorl/cars/dataset2/images'

    pathToSaveImageBoxes1 = '/home/sergorl/cars/car_boxes_1.npy'
    pathToSaveImageBoxes2 = '/home/sergorl/cars/car_boxes_2.npy'

    pathToSaveDataSetDir = '/home/sergorl/cars'

    # parse XMLs and create train and valid sets in pathToSaveDataSetDir
    db1 = DatasetBuilder(pathToXML1,
                         pathToImages1,
                         pathToSaveDataSetDir,
                         pathToSaveImageBoxes1)

    db2 = DatasetBuilder(pathToXML2,
                         pathToImages2,
                         pathToSaveDataSetDir,
                         pathToSaveImageBoxes2)





