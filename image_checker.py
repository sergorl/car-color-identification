import cv2
import os
import numpy as np


colors = {'белый': 'white',
          'черный': 'black',
          'красный': 'red',
          'синий': 'blue',
          'оранжевый': 'orange',
          'коричневый': 'brown',
          'серый': 'grey',
          'темно-серый': 'dark grey',
          'зеленый': 'green'}


def readImageAndDrawBoxes(pathToImage, boxes):

    img = cv2.imread(pathToImage)

    for box in boxes:

        cv2.rectangle(img, (box['xtl'], box['ytl']), (box['xbr'], box['ybr']), (0, 0, 255), 2)

        cv2.putText(img, colors[box['label']],
                    (box['xtl'], box['ytl']),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    2)

        cv2.putText(img, '(' + str(box['xtl']) + ', ' + str(box['ytl']) + ')',
                    (box['xbr'], box['ybr']),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2)

        # cv2.imwrite("my.png", img)

    cv2.imshow("boxed", img)
    cv2.waitKey(0)


if __name__ == '__main__':

    pathToImages1 = '/home/sergorl/cars/dataset1/images'
    pathToImages2 = '/home/sergorl/cars/dataset2/images'

    images = np.load('/home/sergorl/cars/car_boxes_1.npy').item()
    images.update(np.load('/home/sergorl/cars/car_boxes_2.npy').item())

    image = 'vlcsnap-2019-08-07-22h56m06s489.png'
    # image = 'vlcsnap-2019-08-07-23h09m17s466.png'

    pathToErrorImage1 = os.path.join(pathToImages1, image)
    pathToErrorImage2 = os.path.join(pathToImages2, image)

    if os.path.exists(pathToErrorImage1):
        readImageAndDrawBoxes(pathToErrorImage1, images[image])
    elif os.path.exists(pathToErrorImage2):
        readImageAndDrawBoxes(pathToErrorImage2, images[image])

