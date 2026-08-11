const std = @import("std");

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

    var block_types: [num_layers]type = undefined;
    var flat_types: [num_layers]type = undefined;
    for (0..num_layers) |i| {
        const in = dimensions[i];
        const out = dimensions[i + 1];

        block_types[i] = [out][in]f32;
        flat_types[i] = [out]f32;
    }

    const Blocks = @Tuple(&block_types);
    const Flats = @Tuple(&flat_types);

    switch (optim_config.optimizer) {
        .SGD => return struct {
            model_ptr: *MLP(mlp_config),
            lr: f32,
            momentum: ?f32,
            weight_velocities: Blocks,
            bias_velocities: Flats,

            const Self = @This();
            pub fn init(model_ptr: *MLP(mlp_config), lr: f32, momentum: ?f32) OptimizerError!Self {
                if (lr < 0) return OptimizerError.InvalidLearningRate;
                if (momentum) |val| if (val < 0) return OptimizerError.InvalidMomentum;

                var weight_velocities: Blocks = undefined;
                var bias_velocities: Flats = undefined;
                inline for (0..num_layers) |i| {
                    const out = dimensions[i + 1];

                    for (0..out) |j| {
                        weight_velocities[i][j] = @splat(0);
                        bias_velocities[i][j] = 0;
                    }
                }

                return .{
                    .model_ptr = model_ptr,
                    .lr = lr,
                    .momentum = momentum,
                    .weight_velocities = weight_velocities,
                    .bias_velocities = bias_velocities,
                };
            }

            pub fn zero_grad(self: Self) void {
                inline for (0..num_layers) |i| {
                    const model_ptr = self.model_ptr;
                    const in = dimensions[i];
                    const out = dimensions[i + 1];

                    model_ptr.layers[i].weight_grads = [_][in]f32{@splat(0)} ** out;
                    model_ptr.layers[i].bias_grads = @splat(0);
                }
            }

            pub fn step(self: *Self) void {
                if (self.momentum) |momentum| {
                    inline for (0..num_layers) |i| {
                        const model_ptr = self.model_ptr;
                        const lr = self.lr;
                        const in = dimensions[i];
                        const out = dimensions[i + 1];

                        for (0..out) |j| {
                            for (0..in) |k| {
                                const prev_weight_velocity = self.weight_velocities[i][j][k];
                                const weight_grad = model_ptr.layers[i].weight_grads[j][k];

                                self.weight_velocities[i][j][k] = momentum * prev_weight_velocity + lr * weight_grad;

                                const new_weight_velocity = self.weight_velocities[i][j][k];
                                const prev_weight = model_ptr.layers[i].weights[j][k];

                                model_ptr.layers[i].weights[j][k] = prev_weight - new_weight_velocity;
                            }

                            const prev_bias_velocity = self.bias_velocities[i][j];
                            const bias_grad = model_ptr.layers[i].bias_grads[j];

                            self.bias_velocities[i][j] = momentum * prev_bias_velocity + lr * bias_grad;

                            const new_bias_velocity = self.bias_velocities[i][j];
                            const prev_bias = model_ptr.layers[i].biases[j];

                            model_ptr.layers[i].biases[j] = prev_bias - new_bias_velocity;
                        }
                    }
                } else {
                    inline for (0..num_layers) |i| {
                        const model_ptr = self.model_ptr;
                        const lr = self.lr;
                        const in = dimensions[i];
                        const out = dimensions[i + 1];

                        for (0..out) |j| {
                            for (0..in) |k| model_ptr.layers[i].weights[j][k] -= lr * model_ptr.layers[i].weight_grads[j][k];
                            model_ptr.layers[i].biases[j] -= lr * model_ptr.layers[i].bias_grads[j];
                        }
                    }
                }
            }
        },
        .RMSprop => return struct { lr: f32 },
        .Adam => return struct { lr: f32 },
    }
}
