import sys
import numpy as np
import struct
import matplotlib.pyplot as plt
import math
from tqdm import tqdm, trange

def LoadImages(filepath):
    with open(filepath,'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        data = data.reshape((size, nrows, ncols))

    return data

def LoadLabels(filepath):
    with open(filepath,'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        # data = data.reshape((size,))

    return data

def OutputPosterior(posterior):
    n_class = len(posterior)
    highest_prob = 1
    prediction = 0

    print("Posterior (in log scale):")

    for label in range(n_class):
        prob = posterior[label]
        print(label, ": ", prob)
        if prob < highest_prob:
            highest_prob = prob
            prediction = label
    
    return prediction


def Discrete(x_train, y_train, x_test, y_test):
    n_train, n_test = len(x_train), len(x_test)
    n_row, n_col = x_train[0].shape
    classes = sorted(np.unique(y_test))
    n_class = len(classes)
    n_bin = 32
    peudocount = 0.00000001
    
    prior = np.zeros(n_class) # compute prior prob for each label(theda)
    likelihood = np.zeros((10, n_row*n_col, n_bin)) # likelihood[label][pixels][bin]

    for i in trange(n_train):
        prior[y_train[i]] += 1
        for row in range(n_row):
            for col in range(n_col):
                bin = x_train[i][row][col] // 8
                likelihood[y_train[i]][row*n_col + col][bin] += 1

    for label in range(n_class):
        for pixel in range(n_row*n_col):
            for bin in range(n_bin):
                if likelihood[label][pixel][bin] == 0:
                    likelihood[label][pixel][bin] = peudocount # let the peudocount to be the minimum of possible number of bin
                else:
                    likelihood[label][pixel][bin] /= prior[label] # number of each bin need to be divided to number of data of its label

    prior /= n_train
    
    error = 0
    for i in range(n_test):
        posterior = np.zeros(n_class)
        for label in range(n_class):
            posterior[label] = math.log(prior[label])
            for row in range(n_row):
                for col in range(n_col):
                    bin = x_test[i][row][col] // 8
                    posterior[label] += math.log(likelihood[label][row*n_col + col][bin]) # don't need to compute prob of marginal prob
        
        posterior /= sum(posterior)
        prediction = OutputPosterior(posterior)
        print("Prediction: ", prediction, ", Ans: ", y_test[i])
        print()

        if prediction != y_test[i]:
            error += 1

    error /= n_test

    # Print out the imagination of numbers in your Bayes classifier
    print("Imagination of numbers in Bayesian classifier:")

    for label in range(n_class):
        print()
        print(label, ":")
        for row in range(n_row):
            for col in range(n_col):
                # max of bins of likelihood[label][pixel] happens in 0 ~ 15 -> 0, 16 ~ 31 -> 1
                if np.argmax(likelihood[label][row*n_col + col]) <= 15: print("0 ", end="")
                else:                                                   print("1 ", end="")
            
            print()

    print("Error rate: ", error)

    return

def Continuous(x_train, y_train, x_test, y_test):
    return

if __name__ == "__main__":
    if len(sys.argv) < 2:
            print('no argument')
            sys.exit()

    train_images_filepath = sys.argv[1]
    train_labels_filepath = sys.argv[2]
    test_images_filepath = sys.argv[3]
    test_labels_filepath = sys.argv[4]
    toggle = int(sys.argv[5])

    x_train = LoadImages(train_images_filepath)
    x_test = LoadImages(test_images_filepath)
    y_train = LoadLabels(train_labels_filepath)
    y_test = LoadLabels(test_labels_filepath)

    # plt.imshow(x_test[0,:,:], cmap='gray')
    # plt.show()
    # print(y_test[0])

    if toggle == 0: Discrete(x_train, y_train, x_test, y_test)
    else:           Continuous(x_train, y_train, x_test, y_test)

    
    
    