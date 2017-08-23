import numpy as np
import os

from six.moves import cPickle as pickle
import cv2

dirs = ['Dataset/yawnMouth', 'Dataset/normalMouth']
countYawn = 40
countNormal = 34


def generate_dataset():
    '''countYawn = 0
    countNormal = 0
    maxY = 0
    maxX = 0
    minX = 1000
    minY = 1000
    pos = 0
    for dir in dirs:
        for filename in os.listdir(dir):
            if filename.endswith('.png'):
                im = cv2.imread(dir + '/' + filename)
                maxX = max(maxX, im.shape[0])
                minX = min(minX, im.shape[0])
                maxY = max(maxY, im.shape[1])
                minY = min(minY, im.shape[1])
                if pos == 0:
                    countYawn +=1
                else:
                    countNormal += 1
        pos += 1
    print(minX, maxX, minY, maxY, countYawn, countNormal)'''
    maxX = 60
    maxY = 60
    dataset = np.ndarray([countYawn + countNormal, maxY, maxX, 1], dtype='float32')
    i = 0
    j = 0
    pos = 0
    for dir in dirs:
        for filename in os.listdir(dir):
            if filename.endswith('.png'):
                im = cv2.imread(dir + '/' + filename)
                im = cv2.resize(im, (maxX, maxY))
                im = np.dot(np.array(im, dtype='float32'), [[0.2989], [0.5870], [0.1140]]) / 255
                # print(i)
                dataset[i, :, :, :] = im[:, :, :]
                i += 1
        if pos == 0:
            labels = np.ones([i, 1], dtype=int)
            j = i
            pos += 1
        else:
            labels = np.concatenate((labels, np.zeros([i - j, 1], dtype=int)))
    return dataset, labels


dataset, labels = generate_dataset()
print("Total = ", len(dataset))

totalCount = countYawn + countNormal
split = int(countYawn * 0.8)
splitEnd = countYawn
split2 = countYawn + int(countNormal * 0.8)

train_dataset = dataset[:split]
train_labels = np.ones([split, 1], dtype=int)
test_dataset = dataset[split:splitEnd]
test_labels = np.ones([splitEnd - split, 1], dtype=int)

train_dataset = np.concatenate((train_dataset, dataset[splitEnd:split2]))
train_labels = np.concatenate((train_labels, np.zeros([split2 - splitEnd, 1], dtype=int)))
test_dataset = np.concatenate((test_dataset, dataset[split2:]))
test_labels = np.concatenate((test_labels, np.zeros([totalCount - split2, 1], dtype=int)))

pickle_file = 'yawn_mouths.pickle'

try:
    f = open(pickle_file, 'wb')
    save = {
        'train_dataset': train_dataset,
        'train_labels': train_labels,
        'test_dataset': test_dataset,
        'test_labels': test_labels,
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)
