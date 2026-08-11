const mlp = @import("mlp");
const MLPConfig = mlp.MLPConfig;
const MLP = mlp.MLP;

pub const OptimizerAlgo = enum { SGD, RMSprop, Adam };
pub const OptimizerConfig = struct { mlp_config: MLPConfig, optimizer: OptimizerAlgo };
pub const OptimizerError = error{ InvalidLearningRate, InvalidMomentum };

pub fn Optimizer(comptime optim_config: OptimizerConfig) type {
    const mlp_config = optim_config.mlp_config;

    mlp_config.check();
    const num_layers = mlp_config.outs.len;
    const dimensions = [1]usize{mlp_config.in} ++ mlp_config.outs;

    switch (optim_config.optimizer) {
        .SGD => return struct {
            model_ptr: *MLP(mlp_config),
            lr: f32,
            momentum: ?f32,

            const Self = @This();
            pub fn init(model_ptr: *MLP(mlp_config), lr: f32, momentum: ?f32) OptimizerError!Self {
                if (lr < 0) return OptimizerError.InvalidLearningRate;
                if (momentum) |val| if (val < 0) return OptimizerError.InvalidMomentum;

                return .{ .model_ptr = model_ptr, .lr = lr, .momentum = momentum };
            }

            pub fn zero_grad(self: Self) void {
                inline for (0..num_layers) |i| {
                    const model_ptr = self.model_ptr;
                    const in = dimensions[i];
                    const out = dimensions[i + 1];

                    model_ptr.layers[i].weights_grad = [_][in]f32{@splat(0)} ** out;
                    model_ptr.layers[i].biases_grad = @splat(0);
                }
            }

            pub fn step(self: Self) void {
                inline for (0..num_layers) |i| {
                    const model_ptr = self.model_ptr;
                    const lr = self.lr;
                    const in = dimensions[i];
                    const out = dimensions[i + 1];

                    if (self.momentum) |momentum| {
                        _ = momentum;
                    } else {
                        for (0..out) |j| {
                            for (0..in) |k| model_ptr.layers[i].weights[j][k] -= lr * model_ptr.layers[i].weights_grad[j][k];
                            model_ptr.layers[i].biases[j] -= lr * model_ptr.layers[i].biases_grad[j];
                        }
                    }
                }
            }
        },
        .RMSprop => return struct { lr: f32 },
        .Adam => return struct { lr: f32 },
    }
}
