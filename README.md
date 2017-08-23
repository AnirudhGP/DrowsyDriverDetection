# DrowsyDriverDetection
Drowsy driver detection using Keras and convolution neural networks.

## Datasets:

Eye dataset: http://parnec.nuaa.edu.cn/xtan/data/datasets/dataset_B_Eye_Images.rar

Yawn dataset: http://www.discover.uottawa.ca/images/files/external/YawDD_Dataset/YawDD.rar

Credits: [Eye dataset](http://parnec.nuaa.edu.cn/xtan/data/ClosedEyeDatabases.html),[Yawn dataset](http://www.eecs.uottawa.ca/~shervin/yawning)

Note: The datasets are extracted and pickled for easier usage, the pickled files are- `clode_eyes.pickle`, `open_eyes.pickle`, `yawn_mouths.pickle`

## Files included:

`EyePreprocess.py` and `YawnPreprocess.py` : Preprocesses the data by converting the images to grayscale and dividing them into training and testing sets

`EyesCNN.py` and `YawnCNN.py` : Trains a CNN based on the training data.

`EyeDetect.py` and `FaceDetect.py` : Simple eyes and face detection code. Uses a 16-layer cascade instead of the traditional one since the original one was not able to detect faces properly.

Pickle files contain the preprocessed datasets for closed eyes, open eyes and yawns.


