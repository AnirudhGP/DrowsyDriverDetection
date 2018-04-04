# DrowsyDriverDetection
Drowsy driver detection using Keras and convolution neural networks.

## Datasets:

Eye dataset(Not available anymore): http://parnec.nuaa.edu.cn/xtan/data/datasets/dataset_B_Eye_Images.rar

Yawn dataset: http://www.discover.uottawa.ca/images/files/external/YawDD_Dataset/YawDD.rar

Credits: [Eye dataset](http://parnec.nuaa.edu.cn/xtan/data/ClosedEyeDatabases.html),[Yawn dataset](http://www.eecs.uottawa.ca/~shervin/yawning)

Note:
Pickle files contain the preprocessed datasets for closed eyes, open eyes and yawns,
the pickled files are- `closed_eyes.pickle`, `open_eyes.pickle`, `yawn_mouths.pickle`.


## Files included:

`eyePreprocess.py` and `yawnPreprocess.py` : Preprocess the data by converting the images to grayscale and dividing them into training and testing sets

`eyesCNN.py` and `yawnCNN.py` : Train a CNN based on the training data.

`Code_archive/eyeDetect.py` and `Code_archive/faceDetect.py` : Simple eyes and face detection code use a 16-layer cascade instead of the traditional one since the original one was not able to detect faces properly.
