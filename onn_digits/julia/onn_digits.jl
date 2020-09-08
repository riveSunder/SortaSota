using PyCall
using Plots
using Images
using MLDatasets
using Zygote
using FFTW
using PaddedViews
using Statistics

@pyimport sklearn.datasets as datasets
@pyimport matplotlib.pyplot as plt

function asm_prop(wavefront, fxx, fyy, my_length=32.0e-3, wavelength=550.0e-9, distance=10.0e-3)
    
    dim_x, dim_y = size(wavefront)[2:3]
    
    if dim_x != dim_y
        println("Wavefront should be square")
    end
    
    px = my_length / dim_x
    
    l2 = (1/wavelength)^2 
    
    #meshgrid(xx,xx)    
    
    q = l2 .- fxx.^2 .- fyy.^2
    
    q = reshape(q, 1, dim_x, dim_y)
    
    # no mutating arrays with Zygote, so make a new array
    q2 = (q.>= 0.0) .* q
           
    h = fftshift(exp.(im * 2 * π * distance * sqrt.(q2)), [2,3])
    
    fd_wavefront = fft(fftshift(wavefront, [2,3]), [2,3])
    
    fd_new_wavefront = h .* fd_wavefront
     
    new_wavefront = ifftshift(ifft(fd_new_wavefront, [1,2]), [1,2]) #[:dim_x,:dim_x]
    
end

function meshgrid(x, y)
    xx = [ii for ii in x, jj in 1:length(y)]
    yy = [jj for ii in 1:length(x), jj in y]
    return xx, yy
end

function kerr_effect(wavefront, n2=1e-20)
   return exp.(im .* abs.(wavefront.^2) * n2) 
end

function softmax(x)
    
    x = x .- maximum(x, dims=2) .+ 1.e-3

    (exp.(x)) ./ sum(exp.(x), dims=[2])
    
end

function get_accuracy(y, pred, boundary=0.5) 
    temp = [elem[2] for elem in argmax(y,dims=2)]
    temp2 = [elem[2] for elem in argmax(pred,dims=2)]
    

    return mean(temp .== temp2) 
end

log_loss = function(y_tgt, pred)
   
    return -(1 / size(y_tgt)[1]) .* sum(y_tgt .* log.(pred) .+ (1.0 .- y_tgt) .* log.(1.0 .- pred))

end

function get_one_hot(tgt_y)
    
    batch_size, num_classes = size(tgt_y)[1], 10
    one_hot_y = zeros(batch_size, num_classes)
    [one_hot_y[hh, tgt_y[hh]+1] = 1.0 for hh in 1:batch_size]
    
    return one_hot_y 
    
end

function multiply_and_propagate(wavefront, phase_slices, fxx, fyy, n2=1e-20)
    
    for slice_idx in 1:length(phase_slices)
    
        wavefront = asm_prop(wavefront .* (phase_slices[slice_idx] .*
                kerr_effect(wavefront, n2)), fxx, fyy)
    
    end
        
    #wavefront = asm_prop(wavefront .* (phase_slices[:2] .* kerr_effect(wavefront, n2)))
    #wavefront = asm_prop(wavefront .* (phase_slices[:3] .* kerr_effect(wavefront, n2)))
    
    
    #end
    
    return wavefront
    
end

function optical_network(in_x, phase_slices, fxx, fyy, n2=1e-20)
    
    dim_n, dim_x, dim_y = size(in_x)
    
    if dim_x != dim_y
        println("Wavefront should be square")
    end
    
    start_x = Int(round(dim_x//2));
    stop_x = Int(round(3dim_x//2))-1;
    start_y = Int(round(dim_y//2));
    stop_y = Int(round(3dim_y//2))-1;
   
    #hard-coded padding, TODO: generalize for variable dimensions
    #wavefront = PaddedView(0.0 + 0.0 * im, data_array, (dim_n, 56,56), (1, 15,15))
    wavefront = in_x # ones(dim_n, dim_x, dim_y) .* exp.(im*2π .* in_x) 
    
    wavefront2 = multiply_and_propagate(wavefront, phase_slices, fxx, fyy, n2)
    
    return wavefront2
    
end

function get_decision_zones(dim_n=4, dim_x=28, dim_y=28, num_classes = 10)
    
    # assuming a 10 class classifier (i.e. a digits dataset)
    xx, yy = meshgrid(1:dim_x, 1:dim_y)
    
    center_coords = zeros(16,2)
    for ii in 1:4
        for jj in 1:4
            center_coords[4*(ii-1) + jj,:] = [(ii*5) + 2, (jj*5) + 2]
        end
    end
    
    #center_coords = center_coords[3:12,:]
    
    decision_zones = zeros(dim_x, dim_y, 10)
    
    dec_r = 2.5 
      
    for ll in 1:10
        
        decision_zone = sqrt.((xx .- center_coords[ll,1]).^2 .+ (yy .- center_coords[ll,2]).^2)
        
        decision_zone[decision_zone .<= dec_r] .= 1.0 
        decision_zone[decision_zone .> dec_r] .*= 0.0 
        
        decision_zones[:,:,ll] .= decision_zone #./ (1.0e-5 .+ decision_zone)
        
    end
    
    return decision_zones
    
end

function get_pred(I, dz)
    
    dim_n = size(I)[1]
    
    
    temp = reshape(I,size(I)[1], size(I)[2], size(I)[3], 1) .* 
        reshape(dz, 1, size(dz)[1], size(dz)[2], size(dz)[3])
    

    pred = reshape(sum(temp, dims=[2,3]), size(I)[1], size(dz)[3])
 
end

function get_onn_loss(in_x, phase_slices, tgt_y, decision_zones, fxx, fyy, n2=1e-20)
    
    #wavefront = optical_network(in_x, phase_slices, fxx, fyy, n2)
    
    #dim_n, dim_x, dim_y = size(wavefront)
    
    #my_length = 32.e-3
    #px = my_length / dim_x
    
    #xx = range(-1/(2*px), stop=1/(2*px) - 1/(dim_x*px), length=dim_x)
    
    #xx, yy = meshgrid(1:dim_x, 1:dim_y)
    
    intensity = get_intensity(in_x, phase_slices, fxx, fyy, n2) #abs.(wavefront.^2)
    
    intensity2 = intensity ./ maximum(intensity, dims=[2,3])
    
    pred = get_pred(intensity2, decision_zones) 
    
    pred_sm = softmax(pred)
    
    loss = log_loss(tgt_y, pred_sm)
    #accuracy = get_accuracy(tgt_y, pred_sm)
    #println(accuracy)
    return loss    
      
end

function get_intensity(in_x, phase_slices, fxx, fyy, n2 = 1e-20)
    
    wavefront = optical_network(in_x, phase_slices, fxx, fyy, n2)
    
    intensity = abs.(wavefront.^2)
    return intensity
    
end



x, y = datasets.load_digits(return_X_y=true)

x = permutedims(x, [3,2,1]);

val_x = reshape(x[1:256,:],256, 8, 8);
val_y = get_one_hot(y[1:256]);

train_x = reshape(x[257:1280, :], 1024, 8, 8);
train_y = get_one_hot(y[256:1280]);

test_x = reshape(x[1281:1797, :], 517, 8, 8);
test_y = get_one_hot(y[1280:1797]);

train_x = PaddedView(0.0, train_x, (1024, 64,64), (1, 31, 31));
test_x = PaddedView(0.0, test_x, (517, 64,64), (1, 31, 31));
val_x = PaddedView(0.0, test_x, (256, 64,64), (1, 31, 31));

batch_size, dim_x, dim_y = 32, 64, 64;



num_slices = 2
phase_slices = [1.0 .* exp.((im * 2π) .* zeros(1,dim_x, dim_y)/100)
    for ii in 1:num_slices]


decision_zones = get_decision_zones(batch_size, dim_x, dim_y, 10)
xx, yy = meshgrid(1:dim_x, 1:dim_y)

my_length = 32.e-3
px = my_length / dim_x

xx = range(-1/(2*px), stop=1/(2*px) - 1/(dim_x*px), length=dim_x)

fxx, fyy = meshgrid(xx,xx)

train_x = ones(1, dim_x, dim_y) .* 
                exp.(im*2π .* train_x) .*
                reshape((1.0 .* sqrt.(fxx.^2 + fyy.^2) .<= 300),1,dim_x,dim_y)

val_x = ones(1, dim_x, dim_y) .* 
                exp.(im*2π .* val_x) .*
                reshape((1.0 .* sqrt.(fxx.^2 + fyy.^2) .<= 300),1,dim_x,dim_y)

test_x = ones(1, dim_x, dim_y) .* 
                exp.(im*2π .* test_x) .*
                reshape((1.0 .* sqrt.(fxx.^2 + fyy.^2) .<= 300),1,dim_x,dim_y)
lr = 1e-3
n2 = 1e-2

println(maximum(fxx))

plt.figure()
plt.subplot(121)
plt.imshow(angle.(train_x[1,:,:]))
plt.subplot(122)
plt.imshow(abs2.(train_x[1,:,:]))
plt.show()

for step in 1:100

    if (step-1) % 1 == 0

        loss =  get_onn_loss(val_x, phase_slices, val_y, decision_zones, fxx, fyy);
        intensity = get_intensity(val_x, phase_slices, fxx, fyy, n2)
        intensity2 = intensity ./ maximum(intensity, dims=[2,3])
        pred = get_pred(intensity2, decision_zones)
        pred_sm = softmax(pred)
        accuracy = get_accuracy(val_y, pred_sm)

        println("step $step, loss = $loss, accuracy = $accuracy"); #, size(d_slices))

    end

    for batch_end_idx in batch_size+1:size(train_x)[1]
        in_x = train_x[batch_end_idx-batch_size:batch_end_idx,:,:]
        tgt_y = train_y[batch_end_idx-batch_size:batch_end_idx,:]

        d_slices = gradient((phase_slices) ->
            get_onn_loss(in_x, phase_slices, tgt_y, decision_zones, fxx, fyy),
            phase_slices);

        for slice_idx in 1:length(phase_slices)
            phase_slices[slice_idx] .*= - exp.(im .* angle.(lr .* d_slices[1][slice_idx]));
        end
    end

end


