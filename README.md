## Car color identification

This project is about identification the colors of cars. It is based on _CNN_ architecture.
It is implemented with using `python 3.7`.


### Python scripts
1. `dataset_builder.py` parses XML annotations and creates images in specific dirs
according `ImageDataGenerator` and `trainDataGen.flow_from_directory(...)`.
More precisely, it cuts off sub images according bounding boxes from XML annotations and puts them in dirs, matching to few colors.  
2. `color_net.py` contains the _CNN_ architecture and the main functions to train net and predict on images.
3. `test_net.py` trains net ant test it making videos from original images and predicted labels.
4. `image_checker.py` is intended for checking annotations - is it bad or good.
It helps to detect wrong labels on the image. 


### Dependencies
1. `Keras 2.0.0`
2. `Tensorflow 1.14.0`
3. `OpenCV 4.1.0`
4. `Numpy 1.16.2`


### Data sets
1. `109 Коридор мойки.zip`
2. `102 Въезд из мойки в ремзону.zip`


### Problems
1. Some colors (green, brown and grey) are missed at both data sets.
2. Some annotations are distorted: bounding boxes of not cars are labeled as cars,
for example: images `vlcsnap-2019-08-07-22h56m06s489.png` and `vlcsnap-2019-08-07-23h09m17s466.png`
contain bad annotations. To check this just run script `image_checker.py`.
3. Versions of python packages above don't allow me to save neural network to use it again.
Apparently saving functions from `model.save()` and `ModelCheckpoint()` don't work.
See many related issues here:
    - https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras/issues/13
    - https://github.com/keras-team/keras/issues/4904

So, to see how net works use script `test_net.py`: it loads `color_weights.hdf5` via `loadWeights()` (or trains the net) and then test it,
producing videos with predicted colors on bounding boxes.
    
