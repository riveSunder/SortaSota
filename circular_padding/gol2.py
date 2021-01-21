import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from gol import get_moore_machine

import time

class Square(nn.Module):

    def __init__(self):
        super(Square, self).__init__()

    def forward(self, x):

        x = torch.relu(x)
        x[x > 1.0] *= 0.0

        return x


class MooreMachine(nn.Module):

    def __init__(self):
        super(MooreMachine, self).__init__()

        
        moore_kernel = torch.tensor([[1.,1.,1.], [1.,0.,1.], [1.,1.,1.]],\
                requires_grad=False)

        my_mode = "circular"

        self.model = nn.Conv2d(1, 1, 3, padding=1,\
                padding_mode=my_mode, bias=False)

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.named_parameters():
            param[1][0] = moore_kernel

    def forward(self, x):

        return self.model(x)



class LifeLike(nn.Module):

    def __init__(self, b=[3], s=[2,3]):
        super(LifeLike, self).__init__()

        self.b = b
        self.s = s
        self.moore = MooreMachine()
        self.activate = Square()

        self.initialize_rules()

    def initialize_rules(self):

        self.survive = nn.Conv2d(1, len(self.s), 1, padding=0, \
                groups=1)

        for param in self.survive.parameters():
            param.requires_grad = False

        for named_param in self.survive.named_parameters():
            if "bias" in named_param[0]:
                for ii in range(len(named_param[1])):
                    named_param[1][ii] = -1.0 * (self.s[ii] - 0.5)
            else: 
                for jj in range(len(named_param[1])):
                    named_param[1][jj] = 1.0 
        

        self.born = nn.Conv2d(1, len(self.b), 1, padding=0,\
                groups=1)

        for param in self.born.parameters():
            param.requires_grad = False

        for named_param in self.born.named_parameters():
            if "bias" in named_param[0]:
                for ii in range(len(named_param[1])):
                    named_param[1][ii] = -1.0 * (self.b[ii]-0.5)
            else: 
                for jj in range(len(named_param[1])):
                    named_param[1][jj] = 1.0 

    def forward(self, grid):

        moore_neighborhood = self.moore(grid)

        newborn_grid = self.activate(self.born(moore_neighborhood))

        survivor_grid = self.activate(self.survive(\
                moore_neighborhood * grid))

        new_grid = torch.sum(torch.cat([newborn_grid, survivor_grid], 1),\
                1, keepdims=True)

        new_grid[new_grid > 0] = 1.0


        return new_grid
        




if __name__ == "__main__":
    # game of life rules
    bb, ss = [3], [2,3]

    grid = torch.zeros(1, 1, 32, 32)

    # make a glider
    grid[0, 0, 8, 16:19] = 1.0
    grid[0, 0, 9, 18] = 1.0
    grid[0, 0, 10, 17] = 1.0

    # make a small spaceship
    grid[0, 0, 16, 16:18] = 1.0
    grid[0, 0, 17, 13:16] = 1.0
    grid[0, 0, 17, 17:19] = 1.0
    grid[0, 0, 18, 13:18] = 1.0
    grid[0, 0, 19, 14:17] = 1.0

    ca = LifeLike(b=bb, s=ss)

    t0 = time.time()
    for step in range(500):

        if(1):
            fig = plt.figure()
            plt.imshow(grid.detach().squeeze().cpu().numpy())
            plt.savefig("frames/step_{}".format(step))

            plt.close(fig)

        grid = ca(grid)

    print("time elapsed", time.time()-t0)

        

