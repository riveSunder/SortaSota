import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch and naive implementations of GoL

def gol_step(grid, n=1, device="cpu"):
    

    if torch.cuda.is_available():
        device = device
    else:
        device = "cpu"
    
    my_kernel = torch.tensor([[1,1,1], [1,0,1], [1,1,1]]).unsqueeze(0).unsqueeze(0).float().to(device)
    old_grid = grid.float().to(device)
  
    while n > 0:
        temp_grid = F.conv2d(old_grid, my_kernel, padding=1)#[:,:,1:-1,1:-1]
        new_grid = torch.zeros_like(old_grid)

        new_grid[temp_grid == 3] = 1 
        new_grid[old_grid*temp_grid == 2] = 1
        
        old_grid = new_grid.clone()

        n -= 1
    #if n > 0:
    #    new_grid = gol_step(new_grid, n=n)
    
    return new_grid.to("cpu")


def gol_loop(grid, n=1):
    
    old_grid = grid.squeeze().int()
    dim_x, dim_y = old_grid.shape
    my_kernel = torch.tensor([[1,1,1], [1,0,1], [1,1,1]]).int()
    
    while n > 0:
        
        new_grid = torch.zeros_like(old_grid)
        temp_grid = torch.zeros_like(old_grid)
        for xx in range(dim_x):
            for yy in range(dim_y):
                temp_sum = 0
                
                y_stop = 3 if yy < (dim_y-1) else -1
                x_stop = 3 if xx < (dim_x-1) else -1
                
                temp_sum = torch.sum(my_kernel[\
                                     1*(not(xx>0)):x_stop,\
                                     1*(not(yy>0)):y_stop] * old_grid[\
                                                    max(0, xx-1):min(dim_x, xx+2),\
                                                    max(0, yy-1):min(dim_y, yy+2)])
                
                temp_grid[xx,yy] = temp_sum

        new_grid[temp_grid == 3] = 1 
        new_grid[old_grid*temp_grid == 2] = 1
        
        old_grid = new_grid.clone()

        n -= 1
    #if n > 0:
    #    new_grid = gol_step(new_grid, n=n)
    
    return new_grid
    
if __name__ == "__main__":
    # run benchmarks

  if(0): 
      # draw glider
      grid = torch.zeros(1,1,2048,2048) #256,256)

      grid[0,0,19,17] = 1
      grid[0,0,18,18] = 1
      grid[0,0,17,18] = 1
      grid[0,0,17,17] = 1
      grid[0,0,17,16] = 1

  grid = 1.0 * (torch.rand(1,1,2048,2048) > 0.50)

  for num_steps in [1, 6, 60, 600, 6000]:
      #grid = 1.0 * (torch.rand(1,1,64,64) > 0.50)

      if num_steps < 601:
          t0 = time.time()     
          grid = gol_loop(grid, n=num_steps)
          t1 = time.time()
          print("time for {} gol_loop steps = {:.2e}".format(num_steps, t1-t0))


      grid = 1.0 * (torch.rand(1,1,256,256) > 0.50)

      t2 = time.time()     
      grid = gol_step(grid, n=num_steps)
      t3 = time.time()
      print("(cpu) time for {} gol steps = {:.2e}".format(num_steps, t3-t2))

      if num_steps < 601:
          print("loop/pytorch = {:.4e}".format((t1-t0) / (t3-t2)))

      grid = 1.0 * (torch.rand(1,1,256,256) > 0.50)

      t4 = time.time()     
      grid = gol_step(grid, n=num_steps, device="cuda")
      t5 = time.time()
      print("(gpu) time for {} gol steps = {:.2e}".format(num_steps, t5-t4))
      if num_steps < 601:
          print("loop/pytorch = {:.4e}, loop/gpu_pytorch = {:.4e} pytorch/gpu_pytorch = {:.4e}"\
                .format((t1-t0) / (t3-t2), (t1-t0) / (t5-t4), (t3-t2) / (t5-t4) ))
      else:
          print("pytorch/gpu_pytorch = {:.4e}".format((t3-t2) / (t5-t4) ))

