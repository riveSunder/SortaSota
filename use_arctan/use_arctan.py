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

class TwoHeadedMLP(nn.Module):

    def __init__(self, act=nn.Tanh()):

        super(TwoHeadedMLP, self).__init__()

        self.act = act
        self.depth_encoder = 5
        self.depth_head = 3
        self.depth_decoder = 5

        self.x_dim = 64
        self.y_dim = 10
        self.h_dim = 32
        self.code_dim = 16

        self.dropout_rate = 0.05
        self.initialize_model()

    def initialize_model(self):

        self.encoder = nn.Sequential(nn.Linear(self.x_dim, self.h_dim),\
                self.act,\
                nn.Dropout(p=self.dropout_rate)\
                )

        for ii in range(1, self.depth_encoder-1):
            self.encoder.add_module("encoder{}".format(ii),\
                    nn.Sequential(nn.Linear(self.h_dim, self.h_dim),\
                    self.act,\
                    nn.Dropout(p=self.dropout_rate)\
                    )\
                    )

        self.encoder.add_module("encoder{}".format(self.depth_encoder-1),\
                nn.Sequential(nn.Linear(self.h_dim, self.code_dim),\
                self.act,\
                nn.Dropout(p=self.dropout_rate)))

        self.decoder = nn.Sequential(nn.Linear(self.code_dim, self.h_dim),\
                self.act,\
                nn.Dropout(p=self.dropout_rate))

        for jj in range(1, self.depth_decoder-1):
            self.decoder.add_module("decoder{}".format(jj),\
                    nn.Sequential(nn.Linear(self.h_dim, self.h_dim),\
                    self.act,\
                    nn.Dropout(p=self.dropout_rate)\
                    )\
                    )

        self.decoder.add_module("decoder{}".format(self.depth_decoder-1),\
                nn.Sequential(nn.Linear(self.h_dim, self.x_dim)))

        self.head = nn.Sequential(nn.Linear(self.code_dim, self.h_dim),\
                self.act,\
                nn.Dropout(p=self.dropout_rate))

        for kk in range(1, self.depth_head-1):
            self.head.add_module("head{}".format(kk),\
                    nn.Sequential(nn.Linear(self.h_dim,self.h_dim),\
                    self.act,\
                    nn.Dropout(p=self.dropout_rate)))

        self.head.add_module("head{}".format(self.depth_head-1),\
                nn.Sequential(nn.Linear(self.h_dim,self.y_dim),\
                nn.LogSoftmax(dim=1)))



    def forward(self, x):
        
        features = self.encoder(x)
        reconstruction = self.decoder(features)
        logits = self.head(features)

        return logits, reconstruction



if __name__ == "__main__":

    dataset = get_sk_digits()

    model = TwoHeadedMLP()

    x = torch.rand(512,64)
    y_rand = torch.rand(512,10)

    y_onehot = torch.zeros(512,10)

    for number, index in enumerate(torch.argmax(y_rand,dim=1)):
        y_onehot[number, index] = 1.0

    y_tensor = torch.argmax(y_rand, dim=1)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    mse_loss = nn.MSELoss()
    nll_loss = nn.NLLLoss()

    for step in range(100):
        model.zero_grad()

        logits, reconstruction = model(x)

        loss = mse_loss(reconstruction, x) + nll_loss(logits, y_tensor)
        loss.backward()



        print("loss at step {} = {:.3e}".format(step, loss))



