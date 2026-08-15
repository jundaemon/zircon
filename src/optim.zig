const std = @import("std");
const math = std.math;
const testing = std.testing;

const mlp = @import("mlp");
const MLPConfig = mlp.MLPConfig;
const MLP = mlp.MLP;

pub const OptimizerAlgo = enum { SGD, RMSprop, Adam };
pub const OptimizerConfig = struct {
    mlp_config: MLPConfig,
    optimizer: OptimizerAlgo,
};
pub const OptimizerError = error{
    InvalidLearningRate,
    InvalidMomentum,
    InvalidDecayRate,
    InvalidEpsilon,
};

const default_lr = 1e-3;
const default_momentum = 0;
const default_decay_rate = 0.99;
const default_epsilon = 1e-8;

pub const SGDOpts = struct {
    lr: f32 = default_lr,
    momentum: f32 = default_momentum,
};
pub const RMSpropOpts = struct {
    lr: f32 = default_lr,
    decay_rate: f32 = default_decay_rate,
    epsilon: f32 = default_epsilon,
};

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
            momentum: f32,
            W_v: ?Blocks,
            B_v: ?Flats,

            const Self = @This();
            /// in opts,
            /// lr determines the step size in the direction of gradient to take
            /// if given, it has to be more than 0, if not given, it defaults to 1e-3
            ///
            /// momentum determines the significance of past velocities in calculating current velocities
            /// if not given, normal stochastic gradient descent is used to update weights and biases
            /// if given and it is more than 0, stochastic gradient descent with momentum is used to update weights and biases
            pub fn init(model_ptr: *MLP(mlp_config), opts: SGDOpts) OptimizerError!Self {
                const lr = opts.lr;
                const momentum = opts.momentum;

                if (lr <= 0) return OptimizerError.InvalidLearningRate;
                if (momentum < 0) {
                    return OptimizerError.InvalidMomentum;
                } else if (momentum > 0) {
                    var W_v: Blocks = undefined;
                    var B_v: Flats = undefined;
                    inline for (0..num_layers) |i| {
                        const out = dimensions[i + 1];

                        for (0..out) |j| {
                            W_v[i][j] = @splat(0);
                            B_v[i][j] = 0;
                        }
                    }

                    return .{
                        .model_ptr = model_ptr,
                        .lr = lr,
                        .momentum = momentum,
                        .W_v = W_v,
                        .B_v = B_v,
                    };
                } else return .{
                    .model_ptr = model_ptr,
                    .lr = lr,
                    .momentum = momentum,
                    .W_v = null,
                    .B_v = null,
                };
            }

            pub fn zero_grad(self: Self) void {
                const model_ptr = self.model_ptr;

                inline for (0..num_layers) |i| {
                    const in = dimensions[i];
                    const out = dimensions[i + 1];

                    model_ptr.layers[i].dL_dW = [_][in]f32{@splat(0)} ** out;
                    model_ptr.layers[i].dL_dB = @splat(0);
                }
            }

            pub fn step(self: *Self) void {
                const model_ptr = self.model_ptr;
                const lr = self.lr;
                const momentum = self.momentum;

                if (momentum > 0) {
                    inline for (0..num_layers) |i| {
                        const in = dimensions[i];
                        const out = dimensions[i + 1];

                        for (0..out) |j| {
                            for (0..in) |k| {
                                const prev_w_v = self.W_v.?[i][j][k];
                                const dL_dw = model_ptr.layers[i].dL_dW[j][k];

                                // v_t = momentum * v_t-1 + lr * dL_dw
                                self.W_v.?[i][j][k] = momentum * prev_w_v + lr * dL_dw;

                                const curr_w_v = self.W_v.?[i][j][k];
                                const prev_w = model_ptr.layers[i].W[j][k];

                                // w_t = w_t-1 - v_t
                                model_ptr.layers[i].W[j][k] = prev_w - curr_w_v;
                            }

                            const prev_b_v = self.B_v.?[i][j];
                            const dL_db = model_ptr.layers[i].dL_dB[j];

                            self.B_v.?[i][j] = momentum * prev_b_v + lr * dL_db;

                            const curr_b_v = self.B_v.?[i][j];
                            const prev_b = model_ptr.layers[i].B[j];

                            model_ptr.layers[i].B[j] = prev_b - curr_b_v;
                        }
                    }
                } else {
                    inline for (0..num_layers) |i| {
                        const in = dimensions[i];
                        const out = dimensions[i + 1];

                        for (0..out) |j| {
                            for (0..in) |k| {
                                // w_t = w_t-1 - lr * dL_dw
                                const dL_dw = model_ptr.layers[i].dL_dW[j][k];
                                model_ptr.layers[i].W[j][k] -= lr * dL_dw;
                            }
                            const dL_db = model_ptr.layers[i].dL_dB[j];
                            model_ptr.layers[i].B[j] -= lr * dL_db;
                        }
                    }
                }
            }
        },
        .RMSprop => return struct {
            model_ptr: *MLP(mlp_config),
            lr: f32,
            decay_rate: f32,
            epsilon: f32,
            W_moving_mean: Blocks,
            B_moving_mean: Flats,

            const Self = @This();
            /// in opts,
            /// lr determines the step size in the direction of gradient to take
            /// if given, it has to be more than 0, if not given, it defaults to 1e-3
            ///
            /// decay_rate controls how significant past squared gradients are compared to the current squared gradients in calculating the moving mean
            /// if given, it has to be more than 0, if not given, it defaults to 0.99
            ///
            /// epsilon is a safety to prevent division by zero errors
            /// if given, it has to be more than 0, if not given, it defaults to 1e-8
            pub fn init(model_ptr: *MLP(mlp_config), opts: RMSpropOpts) OptimizerError!Self {
                const lr = opts.lr;
                const decay_rate = opts.decay_rate;
                const epsilon = opts.epsilon;

                if (lr <= 0) return OptimizerError.InvalidLearningRate;
                if (decay_rate <= 0) return OptimizerError.InvalidDecayRate;
                if (epsilon <= 0) return OptimizerError.InvalidEpsilon;

                var W_moving_mean: Blocks = undefined;
                var B_moving_mean: Flats = undefined;
                inline for (0..num_layers) |i| {
                    const out = dimensions[i + 1];

                    for (0..out) |j| {
                        W_moving_mean[i][j] = @splat(0);
                        B_moving_mean[i][j] = 0;
                    }
                }

                return .{
                    .model_ptr = model_ptr,
                    .lr = lr,
                    .decay_rate = decay_rate,
                    .epsilon = epsilon,
                    .W_moving_mean = W_moving_mean,
                    .B_moving_mean = B_moving_mean,
                };
            }

            pub fn zero_grad(self: Self) void {
                const model_ptr = self.model_ptr;

                inline for (0..num_layers) |i| {
                    const in = dimensions[i];
                    const out = dimensions[i + 1];

                    model_ptr.layers[i].dL_dW = [_][in]f32{@splat(0)} ** out;
                    model_ptr.layers[i].dL_dB = @splat(0);
                }
            }

            pub fn step(self: *Self) void {
                const model_ptr = self.model_ptr;
                const lr = self.lr;
                const decay_rate = self.decay_rate;
                const epsilon = self.epsilon;

                inline for (0..num_layers) |i| {
                    const in = dimensions[i];
                    const out = dimensions[i + 1];

                    for (0..out) |j| {
                        for (0..in) |k| {
                            const prev_w_moving_mean = self.W_moving_mean[i][j][k];
                            const dL_dw = model_ptr.layers[i].dL_dW[j][k];

                            // s_t = decay_rate * s_t-1 + (1 - decay_rate) * (dL_dw)^2
                            self.W_moving_mean[i][j][k] = decay_rate * prev_w_moving_mean + (1 - decay_rate) * math.pow(f32, dL_dw, 2);

                            const curr_w_moving_mean = self.W_moving_mean[i][j][k];
                            const prev_w = model_ptr.layers[i].W[j][k];

                            // w_t = w_t-1 - lr * dL_dw / (sqrt(s_t) + epsilon)
                            model_ptr.layers[i].W[j][k] = prev_w - lr * dL_dw / (math.sqrt(curr_w_moving_mean) + epsilon);
                        }

                        const prev_b_moving_mean = self.B_moving_mean[i][j];
                        const dL_db = model_ptr.layers[i].dL_dB[j];

                        self.B_moving_mean[i][j] = decay_rate * prev_b_moving_mean + (1 - decay_rate) * math.pow(f32, dL_db, 2);

                        const curr_b_moving_mean = self.B_moving_mean[i][j];
                        const prev_b = model_ptr.layers[i].B[j];

                        model_ptr.layers[i].B[j] = prev_b - lr * dL_db / (math.sqrt(curr_b_moving_mean) + epsilon);
                    }
                }
            }
        },
        .Adam => return struct { lr: f32 },
    }
}

test "optim initialization, errors and defaults" {
    const mlp_config = MLPConfig{
        .in = 1,
        .outs = &.{ 2, 3, 4 },
        .f = &.{ .Tanh, .ReLU, .None },
        .seed = 1,
    };
    var model: MLP(mlp_config) = .init();

    const optimizer: Optimizer(.{
        .mlp_config = mlp_config,
        .optimizer = .SGD,
    }) = try .init(&model, .{});
    try testing.expectEqual(default_lr, optimizer.lr);
    try testing.expectEqual(default_momentum, optimizer.momentum);

    const optimizer_2: Optimizer(.{
        .mlp_config = mlp_config,
        .optimizer = .SGD,
    }) = try .init(&model, .{ .lr = 1e-4, .momentum = 0.9 });
    try testing.expectEqual(1e-4, optimizer_2.lr);
    try testing.expectEqual(0.9, optimizer_2.momentum);

    const optimizer_error: OptimizerError!Optimizer(.{
        .mlp_config = mlp_config,
        .optimizer = .SGD,
    }) = .init(&model, .{ .lr = -1 });
    try testing.expectError(OptimizerError.InvalidLearningRate, optimizer_error);

    const optimizer_error_2: OptimizerError!Optimizer(.{
        .mlp_config = mlp_config,
        .optimizer = .SGD,
    }) = .init(&model, .{ .momentum = -1 });
    try testing.expectError(OptimizerError.InvalidMomentum, optimizer_error_2);

    const optimizer_3: Optimizer(.{
        .mlp_config = mlp_config,
        .optimizer = .RMSprop,
    }) = try .init(&model, .{});
    try testing.expectEqual(default_lr, optimizer_3.lr);
    try testing.expectEqual(default_decay_rate, optimizer_3.decay_rate);
    try testing.expectEqual(default_epsilon, optimizer_3.epsilon);

    const optimizer_4: Optimizer(.{
        .mlp_config = mlp_config,
        .optimizer = .RMSprop,
    }) = try .init(&model, .{ .lr = 1e-4, .decay_rate = 0.9, .epsilon = 1e-7 });
    try testing.expectEqual(1e-4, optimizer_4.lr);
    try testing.expectEqual(0.9, optimizer_4.decay_rate);
    try testing.expectEqual(1e-7, optimizer_4.epsilon);

    const optimizer_error_3: OptimizerError!Optimizer(.{
        .mlp_config = mlp_config,
        .optimizer = .RMSprop,
    }) = .init(&model, .{ .lr = -1 });
    try testing.expectError(OptimizerError.InvalidLearningRate, optimizer_error_3);

    const optimizer_error_4: OptimizerError!Optimizer(.{
        .mlp_config = mlp_config,
        .optimizer = .RMSprop,
    }) = .init(&model, .{ .decay_rate = -1 });
    try testing.expectError(OptimizerError.InvalidDecayRate, optimizer_error_4);

    const optimizer_error_5: OptimizerError!Optimizer(.{
        .mlp_config = mlp_config,
        .optimizer = .RMSprop,
    }) = .init(&model, .{ .epsilon = -1 });
    try testing.expectError(OptimizerError.InvalidEpsilon, optimizer_error_5);
}
