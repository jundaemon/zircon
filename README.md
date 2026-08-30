This was a project written for the purpose of learning about weight initialization, forward passes, backpropagation, loss calculation and network optimization. The library has features like:
- Building of Multi Layer Perceptrons
- Various symmetric (Tanh, Sigmoid) and asymmetric (ReLU) activation functions
- Conditional weight initialization (Xavier and He initialization)
- Various optimizers (SGD, SGD with momentum, RMSprop, Adam)
- Pytorch like API for forward passes, backpropagation and optimization
- Saving and loading of model weights

## Usage
You should not use this in production, but if you would like to try it out, first run
```
zig fetch --save git+https://github.com/jundaemon/zircon
```

Then add this into your `build.zig`
```zig
const zc = b.dependency("zc", .{
        .target = target,
        .optimize = optimize,
});

exe.root_module.addImport("zc", zc.module("zc"));
```

## Example
Below is an example of training an MLP to solve XOR, feel free to use this example to try the library out
```zig
const std = @import("std");
const debug = std.debug;
const Io = std.Io;
const math = std.math;

const zc = @import("zc");
const mlp = zc.mlp;
const MLPConfig = mlp.MLPConfig;
const MLP = mlp.MLP;

const function = zc.function;
const Loss = function.Loss;

const optim = zc.optim;
const Optimizer = optim.Optimizer;

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const inputs = [4][2]f32{ .{ 0, 0 }, .{ 0, 1 }, .{ 1, 0 }, .{ 1, 1 } };
    const labels = [4][1]f32{ .{0}, .{1}, .{1}, .{0} };
    const epochs = 30_000;

    const mlp_config = MLPConfig{
        .in = 2,
        .outs = &.{ 2, 1 },
        .f = &.{ .Sigmoid, .None },
        .seed = 1,
    };
    var model: MLP(mlp_config) = .init();

    const loss_fn: Loss(.{
        .mlp_config = mlp_config,
        .loss_function = .BCE,
    }) = .init(&model);
    var optimizer: Optimizer(.{
        .mlp_config = mlp_config,
        .optimizer = .Adam,
    }) = try .init(&model, .{});

    for (0..epochs) |i| {
        for (inputs, labels) |input, label| {
            const pred = model.forward(input);
            const loss = loss_fn.eval(pred, label);
            debug.print("epoch: {}, loss: {}\n", .{ i, loss.item });

            loss.backward();
            optimizer.step();
            optimizer.zero_grad();
        }
    }

    try model.save(io, "example_weights");
    for (inputs) |input| {
        const raw_pred = model.forward(input)[0];
        const pred = 1 / (1 + math.exp(-raw_pred));
        debug.print("\ninput: {any}, predicted: {d:.2}\n", .{ input, pred });
    }
}
```

