This was a project written for the purpose of learning about weight initialization, forward passes, backpropagation, loss calculation and network optimization. The library has features like:
- Various symmetric (Tanh, Sigmoid) and asymmetric (ReLU) activation functions
- Conditional weight initialization (Xavier and He initialization)
- Various optimizers (SGD, SGD with momentum, RMSprop, Adam)
- Pytorch like API for forward passes, loss calculation, backpropagation and optimization
- Saving and loading of model weights
- No heap allocations

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

You can then import the library in `main.zig` through
```zig
const zc = @import("zc");
```

## Example
An example of training an MLP to solve XOR is available [here](https://github.com/jundaemon/zircon/blob/main/src/main.zig), feel free to use this example to try the library out

