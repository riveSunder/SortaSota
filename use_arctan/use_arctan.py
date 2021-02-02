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

class ArcTan(nn.Module):

    def __init__(self):
        super(ArcTan,self).__init__()

    def forward(self, x):

        return torch.arctan(x)

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

        self.dropout_rate = 0.125
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

def get_accuracy(logits, y_tensor):
        
    accuracy = torch.sum(1.*torch.argmax(logits, dim=1) == y_tensor) / len(y_tensor)
                    
    return accuracy

if __name__ == "__main__":

    dataset = get_sk_digits()


    [[train_x, train_y], [val_x, val_y], [test_x, test_y]] = get_sk_digits()
    y_tensor = torch.argmax(torch.Tensor(train_y), dim=1)
    val_y_tensor = torch.argmax(torch.Tensor(val_y), dim=1)

    train_x = torch.Tensor(train_x)
    val_x = torch.Tensor(val_x)

    
    mse_loss = nn.MSELoss()
    nll_loss = nn.NLLLoss()

    max_epochs = 30000
    display_every = 100

    results = {}

    for my_seed in [13, 1337, 42]:
        np.random.seed(my_seed)
        torch.manual_seed(my_seed)

        for act_name, act_fn in zip(["arctan", "tanh", "relu"], [ArcTan(), nn.Tanh(), nn.ReLU()]):
            
            for loss_name, loss_weight in zip(["mse_loss", "nll_loss"], [1.0, 0.0]):

                results[act_name+loss_name + "_loss" + str(my_seed)] = []
                results[act_name+loss_name + "_val_loss" + str(my_seed)] = []

                results[act_name+loss_name + "_accuracy" + str(my_seed)] = []
                results[act_name+loss_name + "_val_accuracy" + str(my_seed)] = []

                model = TwoHeadedMLP(act = act_fn)
                optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

                for epoch in range(max_epochs):
                    model.zero_grad()

                    logits, reconstruction = model(train_x)

                    loss = loss_weight * mse_loss(reconstruction, train_x) \
                            + (1-loss_weight) * nll_loss(logits, y_tensor)
                    loss.backward()
                    optimizer.step()



                    if epoch % display_every == 0:

                        val_logits, val_reconstruction = model(val_x)
                        val_loss = (loss_weight) * mse_loss(val_reconstruction, val_x) \
                                + (1. - loss_weight) * nll_loss(val_logits, val_y_tensor)

                        accuracy = get_accuracy(logits, y_tensor)
                        val_accuracy = get_accuracy(val_logits, val_y_tensor)

                        print("activation: ", act_name, ", ", loss_name, " loss")
                        print("loss at step {} = {:.3e}".format(epoch, loss))
                        print("validation loss at step {} = {:.3e}".format(epoch, val_loss))

                        print("accuracy at step {} = {:.3e}".format(epoch, accuracy))
                        print("validation accuracy at step {} = {:.3e}".format(epoch, val_accuracy))

                        results[act_name+loss_name + "_loss" + str(my_seed)].append(loss)
                        results[act_name+loss_name + "_val_loss" + str(my_seed)].append(val_loss)

                        results[act_name+loss_name + "_accuracy" + str(my_seed)].append(accuracy)
                        results[act_name+loss_name + "_val_accuracy" + str(my_seed)].append(val_accuracy)



                val_logits, val_reconstruction = model(val_x)
                val_loss = mse_loss(val_reconstruction, val_x) + nll_loss(val_logits, val_y_tensor)

                accuracy = get_accuracy(logits, y_tensor)
                val_accuracy = get_accuracy(val_logits, val_y_tensor)

                print(act_name, " ", loss_name)
                print("final results with {} activation".format(act_name))
                print("loss at step {} = {:.3e}".format(epoch, loss))
                print("validation loss at step {} = {:.3e}".format(epoch, val_loss))

                print("accuracy at step {} = {:.3e}".format(epoch, accuracy))
                print("validation accuracy at step {} = {:.3e}".format(epoch, val_accuracy))

                results[act_name+loss_name + "_loss" + str(my_seed)].append(loss)
                results[act_name+loss_name + "_val_loss" + str(my_seed)].append(val_loss)

                results[act_name+loss_name + "_accuracy" + str(my_seed)].append(accuracy)
                results[act_name+loss_name + "_val_accuracy" + str(my_seed)].append(val_accuracy)

    import pdb; pdb.set_trace()

    np.save("./temp_results_2.npy", results, allow_pickle=True)
