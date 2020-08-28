using Zygote
using Statistics
using Plots
using StatsPlots

x = rand(512,2) .> 0.5
y = zeros(512,1) 

for ii = 1:size(y)[1]
    y[ii] = xor(x[ii,1], x[ii,2])
end

x = x + randn(512,2) / 10


wxh = randn(2,4) / 8
why = randn(4,1) / 4

sigmoid(x) = 1 ./ (1 .+ exp.(-x))
f(x, wxh, why) = sigmoid(sigmoid(x * wxh) * why)

log_loss = function(y, pred)
    return -(1 / size(y)[1]) .* sum(y .* log.(pred) .+ (1.0 .- y) .* log.(1.0 .- pred))
end

get_loss = function(x, wxh, why, y, l2=6e-4)

    pred = f(x, wxh, why)
    loss = log_loss(y, pred)
    loss = loss + l2 * (sum(abs.(wxh.^2)) + sum(abs.(why.^2)))
    return loss

end

get_accuracy(y, pred, boundary=0.5) = mean(y .== (pred .> boundary)) 


train = function(x, wxh, why, y, max_steps=1000, lr = 1e-2)
    
    disp_every = max_steps // 100

    losses = zeros(max_steps)
    acc = zeros(max_steps)

    for step = 1:max_steps
        
        pred = f(x, wxh, why)
        loss = log_loss(y, pred)
        losses[step] = loss 
        acc[step] = get_accuracy(y, pred)

        dwxh, dwhy = gradient((wxh, why) -> get_loss(x, wxh, why, y), wxh, why)

        wxh, why = wxh .- lr .* dwxh, why .- lr .* dwhy
        
        
        
        if mod(step, disp_every) == 0
            violin([" "], reshape(wxh,8), ylims=(-8,8.),
                    color = :blue, alpha = 0.5, label="wxh", title="Weights")
            violin!([" "], reshape(why,4), ylims=(-8,8),
                    color = :green, alpha = 0.5, label="why")

            savefig("wts$step.png")
            
            pred = f(x, wxh, why) 
            loss = log_loss(y, pred)
            accuracy = get_accuracy(y, pred)

            println("loss at step $step = $loss, accuracy = $accuracy")


        end

    end
    return wxh, why, losses, acc
end


wxh, why, losses, acc = train(x, wxh, why, y, 35000, 1e-1)

steps = 1:size(losses)[1]
plot(steps, losses, title="Training XOR", label="loss")
plot!(steps, acc, label="accuracy")

savefig("temp.png")
