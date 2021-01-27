import numpy as np

import torch
import torch.nn as nn

import sklearn
import sklearn.datasets as datasets

def get_sk_digits(my_seed=1337):

    # load and prep data
    [x, y] = datasets.load_digits(return_X_y=True)

    np.random.seed(my_seed)
    np.random.shuffle(x)

    np.random.seed(my_seed)
    np.random.shuffle(y)

    # convert target labels to one-hot encoding
    y_one_hot = np.zeros((y.shape[0],10))

    for dd in range(y.shape[0]):
            y_one_hot[dd, y[dd]] = 1.0
                
            # separate training, test and validation data
            test_size = int(0.1 * x.shape[0])

            train_x, train_y = x[:-2*test_size], y_one_hot[:-2*test_size]
            val_x, val_y = x[-2*test_size:-test_size], y_one_hot[-2*test_size:-test_size]
            test_x, test_y = x[-test_size:], y_one_hot[-test_size:]

    
    return [[train_x, train_y], [val_x, val_y], [test_x, test_y]] 


if __name__ == "__main__":

    dataset = get_sk_digits()

    import pdb; pdb.set_trace()

