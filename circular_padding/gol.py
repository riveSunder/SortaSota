import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

def get_moore_machine(circular=False):

    # conv kernel for calculating a Moore neighborhood
    moore_kernel = torch.tensor([[1,1,1], [1,0,1], [1,1,1]])

    if circular:
        my_mode = "circular"
    else:
        my_mode = "zeros"

    model = nn.Conv2d(1, 1, 3, padding=1, padding_mode=my_mode, bias=False)

    for param in model.named_parameters():
        param[1][0].requres_grad = False

        param[1][0] = moore_kernel

    return model

def gol_step(my_grid, neighborhood_conv, steps=1, device="cpu"):

    previous_grid = my_grid.float().to(device)

    while steps > 0:

        temp = neighborhood_conv(previous_grid)

        new_grid = torch.zeros_like(previous_grid)

        new_grid[temp == 3] = 1
        new_grid[previous_grid*temp == 2] = 1

        previous_grid = new_grid.clone()

        steps -= 1

    return new_grid.to("cpu")

if __name__ == "__main__":

    flat = get_moore_machine()
    toroid = get_moore_machine(circular=True)

    random_grid = torch.randn(64,64) > 0.0

    glider_grid = torch.zeros(64,64)

    glider_grid[32, 32:35] = 1.0
    glider_grid[33, 34] = 1.0
    glider_grid[34, 33] = 1.0

    for my_starting_grid, my_name in zip([random_grid, glider_grid],\
            ["random", "glider"]):

        my_grid = my_starting_grid.clone().unsqueeze(0).unsqueeze(0)


        for step in range(300):

            my_grid = gol_step(my_grid, flat)

            fig = plt.figure()
            plt.imshow(my_grid[0,0,:,:])
            plt.savefig("./frames/flat_gol_{}{}.png".format(my_name, step))
            plt.close(fig)


        my_grid = my_starting_grid.clone().unsqueeze(0).unsqueeze(0)

        for step in range(300):

            my_grid = gol_step(my_grid, toroid)

            fig = plt.figure()
            plt.imshow(my_grid[0,0,:,:])
            plt.savefig("./frames/toroid_gol_{}{}.png".format(my_name, step))
            plt.close(fig)

    print("all done")
