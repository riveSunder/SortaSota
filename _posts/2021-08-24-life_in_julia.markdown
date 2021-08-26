---
layout: post
title: Life-Like Execution Speeds with Julia and PyTorch
date: 2021-08-24 00:00:00 +0000
--- 

# Hello

<div align="center">
<img src="/SortaSota/assets/life_like/glider_animation.gif" width=50%>
</div>


I've been tinkering with the Julia programming language lately, with the aim of determining suitability for developmental machine learning and biologically inspired computing. Along those lines, in this post we'll consider the execution speed of the Julia language for simulating Life-like cellular automata (CA). If you're in a hurry, I'll let you know that I didn't see consistently faster execution times with Julia than with a cellular automata simulator written in PyTorch under a variety of conditions. In fact, Julia was consistently slower in a variety of configurations, ranging from 20% to about 20 times slower depending on parameters. On the other hand, I spent a fair amount of time working on the CA simulator and there are likely several low-hanging fruit for speeding things up in the Julia implementation. In the Python baseline I use PyTorch convolutions and take advantage of the significant development effort that's gone in to optimizing execution speed for deep learning, while in July I use the `FFTW.jl` library to perform convolutions via the convolution theorem of the Fourier transform. 

Programmers used to working with Python will undoubtedly notice several pain points when using Julia. Julia uses just-in-time (JIT) compilation to build a lower-level assembly language version of Julia code. While this can generate substantial speedups compared to similar code executed in Python, the lag associated with JIT will be noticeable in a Python-esque development workflow. Other differences that are readily noticeable early on will be that ranges are inclusive on both ends, and indexing begins at 1 [:/]. But I'll reserve any judgement or a more formal discussion of the languages pros and cons until I'm more familiar with it. 

One thing also striking as a bit dated is the name of the language. Why do people so often call their technology by human names? Is it because deep down they want an anthropomorphized machine servant to do their bidding? Anyway, I'll be comparing an implementation of a totalistic cellular automata simulator written in Julia to another CA simulator project I've been working on, named Cellular Automata Reinforcement Learning Environment, or [CARLE](https:/github.com/rivesunder/carle).

The Game of Life benchmark I am using in this write-up consists of 1 to 10000 steps of a glider in Conway's Game of Life with grid dimensions of 64, 128, or 256 cells wide and tall. In the first case I ran the simulation on a laptop CPU.  

<div align="center">
<img src="/SortaSota/assets/life_like/laptop_julia.png" width=65%>
<br>
1000-step performance simulating Conway's Game of Life with Julia with `FFTW.set_num_threads(n)` set to 1, 2, or 4 threads. Test performed on a laptop with a 4-core Intel i5 processor. 
</div>

<div align="center">
<img src="/SortaSota/assets/life_like/laptop_carle.png" width=65%>
<br>
1000-step performance simulating Conway's Game of Life with CARLE. Test performed on the same laptop with a 4-core Intel i5 processor. 
</div>

I also tried running the benchmark on a more powerful desktop, while still only utilizaing the CPU. (CARLE is actually capable of more than 100,000 grid updates per second when taking advantage of both GPU acceleration and environment vectorization, but we're not ready to make that comparison just yet). Results on the desktop CPU (24-core AMD Threadripper 3960x) followed the same pattern as we saw on the laptop. 

<div align="center">
<img src="/SortaSota/assets/life_like/desktop_julia.png" width=65%>
<br>
1000-step performance simulating Conway's Game of Life with Julia with `FFTW.set_num_threads(n)` set to 1, 8, or 16 threads. Test performed on a desktop with a 24-core AMD 3960x CPU.  
</div>

<div align="center">
<img src="/SortaSota/assets/life_like/desktop_carle.png" width=65%>
<br>
1000-step performance simulating Conway's Game of Life with CARLE. Test performed on the same desktop with a 24-core AMD 3960x CPU.
</div>

And that's it. In this cellular automata simulation benchmark, PyTorch thoroughly bested the Julia implementation in all conditions. However, considering that the single-thread performance in Julia was most comparable to PyTorch (at least in the small 64x64 cell grid condition), there should be ample room for improvement. If the Julia implementation can be improved to better take advantage of multithreading I think it will surpass or at least be competitive with CARLE in some conditions. If you'd like to stick around to go through the Julia code, continue reading. 


## Appendum: Julia Implementation Code

One thing that I'm getting used to with Julia is that the language doesn't support classes. Instead you can write functions that operate on structs. The CA universe struct in this case is defined in the following snippet of code.

```
mutable struct Universe 
    born::Array{Int32}
    live::Array{Int32}
    grid::Array{Int32, 2}
end
```

The CA state is described by the rules for cell transitions from 0 -> 1 (born), and from 1 -> 1 (live). All other cells become or stay 0. `grid` is the array that contains the cell states at each time step. 

In general, you can build a Life-like CA simulator with just two steps: 1) calculate the ([Moore](https://en.wikipedia.org/wiki/Moore_neighborhood)) neighborhood for each cell location and 2) update the cell states based on their current state and that of their neighbors. Calculating the Moore neighborhood is easily accomplished with a convolution operation, and for that I used the Fourier transform method. 

```
function ft_convolve(grid, kernel)
    
    grid2 = grid 
    
    if size(grid2) != size(kernel)
        padded_kernel = pad_to_2d(kernel, size(grid2))
    else
        padded_kernel = kernel
    end
    
    abs.(FFTW.ifftshift(FFTW.ifft(FFTW.fft(FFTW.fftshift(grid)) .* FFTW.fft(padded_kernel) ) ))

    convolved = round.(FFTW.ifftshift(abs.(FFTW.ifft(
                    FFTW.fft(FFTW.fftshift(grid2)) .* 
                    FFTW.fft(FFTW.fftshift(padded_kernel)) ))) )

    
    return convolved 
end
``` 

The fftshifts you see in the function above are necessary to map the convolutions to a toroidal manifold. If you've ever play an old school video game like "Pac-Man" you probably have a sense of how this works: something disappearing of of one edge will re-enter the grid on the opposite edge. In PyTorch the same effect is accomplished by setting the padding `mode` to `circular`. `pad_to_2d` is a function that enlarges the convolution kernel to match the dimensions of the CA grid, so that the sizes of the Fourier-transformed grid and kernel match and point-wise multiplication can be performed. Multiplication in the Fourier domain is transforms to convolutions in the spatial domain, so after taking the inverse FFT the result is the grid convolved with the kernel.  

```
function pad_to_2d(kernel, dims)

    padded = zeros(dims)
    mid_x = Int64(round(dims[1] / 2))
    mid_y = Int64(round(dims[2] / 2))
    mid_k_x = Int64(round(size(kernel)[1] / 2))
    mid_k_y = Int64(round(size(kernel)[2] / 2))
    
    start_x = mid_x - mid_k_x
    start_y = mid_y - mid_k_y
    
    padded[2+mid_x-mid_k_x:mid_x-mid_k_x + size(kernel)[1]+1,
            2+mid_y-mid_k_y:mid_y-mid_k_y + size(kernel)[2]+1] = kernel
    
    #padded[1:size(kernel)[1], 1:size(kernel)[2]] = copy(kernel)

    return padded
end
```

The next step is to apply cell updates according to the B/S rules applied to the state of each cell and its neighbor. I've left a few different methods of doing this as comments in the function below. In particular I thought that using the `reduce` trick would offer a speed advantage, as it led to a 2 to 3 times speed-up when I used it in CARLE. That wasn't the case in Julia, however, and the uncommented looped logic you see before you was the fastest this time. 

```
function ca_update(grid, rules)

    # Moore neighborhood kernel
    kernel = ones(Int32, 3,3)
    kernel[2, 2] = 0
    
    moore_grid = ft_convolve(grid, kernel)
    
    new_grid = zeros(size(moore_grid))
   
    #my_fn(a,b) = a + b
    #born = reduce(my_fn, [elem .== moore_grid for elem in rules[1]])
    #live = reduce(my_fn, [elem .== moore_grid for elem in rules[2]])

    #new_grid = grid .* born + (1 .- grid) * live
 
    for birth in rules[1]
        #new_grid[(round.(moore_grid .- birth) .== 0.0) .& (grid .!= 1)] .= 1
        new_grid[((moore_grid .== birth) .& (grid .!= 1))] .= 1
    end

    for survive in rules[2]
        #new_grid[(round.(moore_grid .- survive) .== 0.0) .& (grid .== 1)] .= 1
        new_grid[((moore_grid .== survive) .& (grid .== 1))] .= 1
    end

    return new_grid 
end
```

Last and of little importance is a small convenience function for running multiple updates.

```
function ca_steps(universe::Universe, steps::Int64)

    for ii = 1:steps
        universe.grid = ca_update(universe.grid, [universe.born, universe.live]) 
    end
```
end

Those are the operations I used to build a CA simulator in Julia. If you'd like to look at the benchmark code and the raw results, check out the [notebooks](https://github.com/riveSunder/life_like/tree/master/notebooks). Thanks for your attention, and have a nice day in whatever universe you're currently inhabiting. 

