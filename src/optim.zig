const std = @import("std");
const math = std.math;

const mlp = @import("mlp.zig");
const MLPConfig = mlp.MLPConfig;
const MLP = mlp.MLP;

pub const OptimizerAlgo = enum { SGD, RMSprop, Adam };
/// configurations for Optimizer
///
/// properties:
/// mlp_config -> reference MLP architecture to build struct around
/// optimizer -> user selected optimizer algorithm
pub const OptimizerConfig = struct {
    mlp_config: MLPConfig,
    optimizer: OptimizerAlgo,
};
pub const OptimizerError = error{
    InvalidLearningRate,
    InvalidMomentum,
    InvalidAlpha,
    InvalidBeta,
    InvalidEpsilon,
};

/// options for if user selects stochastic gradient descent as optimizer algorithm
pub const SGDOpts = struct {
    lr: f32 = 1e-3,
    momentum: f32 = 0,
};
/// options for if user selects root mean square propagation as optimizer algorithm
pub const RMSpropOpts = struct {
    lr: f32 = 1e-2,
    alpha: f32 = 0.99,
    epsilon: f32 = 1e-8,
};
/// options for if user selects adaptive momentum estimation as optimizer algorithm
pub const AdamOpts = struct {
    lr: f32 = 1e-3,
    beta: f32 = 0.9,
    beta_2: f32 = 0.999,
    epsilon: f32 = 1e-8,
};

/// uses the model architecture to build an interface for model optimization
///
/// arguments:
/// optim_config -> comptime configurations containing data like parent MLP architecture and user selected optimizer algorithm
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
            /// if .SGD was selected as the optimizer algorithm, this initializes and returns a struct instance
            ///
            /// arguments:
            /// model_ptr -> pointer to the parent MLP, used in .zero_grad and .step
            /// opts -> user passed options for stochastic gradient descent
            ///
            /// in opts,
            /// momentum determines the significance of past velocities when calculating the current velocity
            /// if momentum is not given, then properties .W_v and .B_v won't be instantiated and stochastic gradient descent is performed in .step
            /// but if momentum is given, then properties .W_v and .B_v are initialized as 0 and stochastic gradient descent with momentum is performed in .step
            ///
            /// if lr or momentum is given, then they have to be more than 0 else an OptimizerError is returned
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

            /// resets all gradients in the parent model to 0
            pub fn zero_grad(self: Self) void {
                const model_ptr = self.model_ptr;

                inline for (0..num_layers) |i| {
                    const in = dimensions[i];
                    const out = dimensions[i + 1];

                    model_ptr.layers[i].dL_dW = [_][in]f32{@splat(0)} ** out;
                    model_ptr.layers[i].dL_dB = @splat(0);
                }
            }

            /// uses the calculated gradients from loss.backward() to update the weights and biases within the model
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
            alpha: f32,
            epsilon: f32,
            W_moving_mean: Blocks,
            B_moving_mean: Flats,

            const Self = @This();
            /// if .RMSprop was selected as the optimizer algorithm, this initializes and returns a struct instance
            ///
            /// arguments:
            /// model_ptr -> pointer to the parent MLP, used in .zero_grad and .step
            /// opts -> user pass options for root mean squared propagation
            ///
            /// in opts,
            /// alpha determines the significance of past squared gradients compared to the current squared gradients when calculating the moving mean
            /// epsilon prevents division by zero errors
            ///
            /// if lr, alpha or epsilon is given, then they have to be more than 0 else an OptimizerError is returned
            pub fn init(model_ptr: *MLP(mlp_config), opts: RMSpropOpts) OptimizerError!Self {
                const lr = opts.lr;
                const alpha = opts.alpha;
                const epsilon = opts.epsilon;

                if (lr <= 0) return OptimizerError.InvalidLearningRate;
                if (alpha <= 0) return OptimizerError.InvalidAlpha;
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
                    .alpha = alpha,
                    .epsilon = epsilon,
                    .W_moving_mean = W_moving_mean,
                    .B_moving_mean = B_moving_mean,
                };
            }

            /// resets all gradients in the parent model to 0
            pub fn zero_grad(self: Self) void {
                const model_ptr = self.model_ptr;

                inline for (0..num_layers) |i| {
                    const in = dimensions[i];
                    const out = dimensions[i + 1];

                    model_ptr.layers[i].dL_dW = [_][in]f32{@splat(0)} ** out;
                    model_ptr.layers[i].dL_dB = @splat(0);
                }
            }

            /// uses the calculated gradients from loss.backward() to update the weights and biases within the model
            pub fn step(self: *Self) void {
                const model_ptr = self.model_ptr;
                const lr = self.lr;
                const alpha = self.alpha;
                const epsilon = self.epsilon;

                inline for (0..num_layers) |i| {
                    const in = dimensions[i];
                    const out = dimensions[i + 1];

                    for (0..out) |j| {
                        for (0..in) |k| {
                            const prev_w_moving_mean = self.W_moving_mean[i][j][k];
                            const dL_dw = model_ptr.layers[i].dL_dW[j][k];

                            // s_t = alpha * s_t-1 + (1 - alpha) * (dL_dw)^2
                            self.W_moving_mean[i][j][k] = alpha * prev_w_moving_mean + (1 - alpha) * math.pow(f32, dL_dw, 2);

                            const curr_w_moving_mean = self.W_moving_mean[i][j][k];
                            const prev_w = model_ptr.layers[i].W[j][k];

                            // w_t = w_t-1 - lr * dL_dw / (sqrt(s_t) + epsilon)
                            model_ptr.layers[i].W[j][k] = prev_w - lr * dL_dw / (math.sqrt(curr_w_moving_mean) + epsilon);
                        }

                        const prev_b_moving_mean = self.B_moving_mean[i][j];
                        const dL_db = model_ptr.layers[i].dL_dB[j];

                        self.B_moving_mean[i][j] = alpha * prev_b_moving_mean + (1 - alpha) * math.pow(f32, dL_db, 2);

                        const curr_b_moving_mean = self.B_moving_mean[i][j];
                        const prev_b = model_ptr.layers[i].B[j];

                        model_ptr.layers[i].B[j] = prev_b - lr * dL_db / (math.sqrt(curr_b_moving_mean) + epsilon);
                    }
                }
            }
        },
        .Adam => return struct {
            model_ptr: *MLP(mlp_config),
            lr: f32,
            beta: f32,
            beta_2: f32,
            epsilon: f32,
            W_moment_estimate: Blocks,
            B_moment_estimate: Flats,
            W_moment_estimate_2: Blocks,
            B_moment_estimate_2: Flats,
            t: usize = 1,

            const Self = @This();
            /// if .Adam was selected as the optimizer algorithm, this initializes and returns a struct instance
            ///
            /// arguments:
            /// model_ptr -> pointer to the parent MLP, used in .zero_grad and .step
            /// opts -> user pass options for adaptive momentum estimation
            ///
            /// in opts,
            /// beta is used in the first moment estimate, which is the velocity component of update
            /// beta_2 is used in the second moment estimate, which is the gradient dependent scaling of lr
            /// epsilon prevents division by zero errors
            ///
            /// if lr, beta, beta_2 or epsilon is given, then they have to be more than 0 else an OptimizerError is returned
            pub fn init(model_ptr: *MLP(mlp_config), opts: AdamOpts) OptimizerError!Self {
                const lr = opts.lr;
                const beta = opts.beta;
                const beta_2 = opts.beta_2;
                const epsilon = opts.epsilon;

                if (lr <= 0) return OptimizerError.InvalidLearningRate;
                if (beta <= 0) return OptimizerError.InvalidBeta;
                if (beta_2 <= 0) return OptimizerError.InvalidBeta;
                if (epsilon <= 0) return OptimizerError.InvalidEpsilon;

                var W_moment_estimate: Blocks = undefined;
                var B_moment_estimate: Flats = undefined;
                inline for (0..num_layers) |i| {
                    const out = dimensions[i + 1];

                    for (0..out) |j| {
                        W_moment_estimate[i][j] = @splat(0);
                        B_moment_estimate[i][j] = 0;
                    }
                }

                return .{
                    .model_ptr = model_ptr,
                    .lr = lr,
                    .beta = beta,
                    .beta_2 = beta_2,
                    .epsilon = epsilon,
                    .W_moment_estimate = W_moment_estimate,
                    .B_moment_estimate = B_moment_estimate,
                    .W_moment_estimate_2 = W_moment_estimate,
                    .B_moment_estimate_2 = B_moment_estimate,
                };
            }

            /// resets all gradients in the parent model to 0
            pub fn zero_grad(self: Self) void {
                const model_ptr = self.model_ptr;

                inline for (0..num_layers) |i| {
                    const in = dimensions[i];
                    const out = dimensions[i + 1];

                    model_ptr.layers[i].dL_dW = [_][in]f32{@splat(0)} ** out;
                    model_ptr.layers[i].dL_dB = @splat(0);
                }
            }

            /// uses the calculated gradients from loss.backward() to update the weights and biases within the model
            pub fn step(self: *Self) void {
                const model_ptr = self.model_ptr;
                const lr = self.lr;
                const beta = self.beta;
                const beta_2 = self.beta_2;
                const epsilon = self.epsilon;

                const t: f32 = @floatFromInt(self.t);
                const beta_correction = math.pow(f32, beta, t);
                const beta_2_correction = math.pow(f32, beta_2, t);

                inline for (0..num_layers) |i| {
                    const in = dimensions[i];
                    const out = dimensions[i + 1];

                    for (0..out) |j| {
                        for (0..in) |k| {
                            const prev_w_moment_estimate = self.W_moment_estimate[i][j][k];
                            const prev_w_moment_estimate_2 = self.W_moment_estimate_2[i][j][k];
                            const dL_dw = model_ptr.layers[i].dL_dW[j][k];

                            // m_t = beta * m_t-1 + (1 - beta) * dL_dw
                            self.W_moment_estimate[i][j][k] = beta * prev_w_moment_estimate + (1 - beta) * dL_dw;
                            // v_t = beta_2 * v_t-1 + (1 - beta_2) * (dL_dw)^2
                            self.W_moment_estimate_2[i][j][k] = beta_2 * prev_w_moment_estimate_2 + (1 - beta_2) * math.pow(f32, dL_dw, 2);

                            // bias corrected moment estimates: m_cap_t = m_t / (1 - beta^t) | v_cap_t = v_t / (1 - beta_2^t)
                            const bc_w_moment_estimate = self.W_moment_estimate[i][j][k] / (1 - beta_correction);
                            const bc_w_moment_estimate_2 = self.W_moment_estimate_2[i][j][k] / (1 - beta_2_correction);
                            const prev_w = model_ptr.layers[i].W[j][k];

                            // w_t = w_t-1 - lr / (sqrt(v_cap_t) + epsilon) * m_cap_t
                            model_ptr.layers[i].W[j][k] = prev_w - lr / (math.sqrt(bc_w_moment_estimate_2) + epsilon) * bc_w_moment_estimate;
                        }

                        const prev_b_moment_estimate = self.B_moment_estimate[i][j];
                        const prev_b_moment_estimate_2 = self.B_moment_estimate_2[i][j];
                        const dL_db = model_ptr.layers[i].dL_dB[j];

                        self.B_moment_estimate[i][j] = beta * prev_b_moment_estimate + (1 - beta) * dL_db;
                        self.B_moment_estimate_2[i][j] = beta_2 * prev_b_moment_estimate_2 + (1 - beta_2) * math.pow(f32, dL_db, 2);

                        const bc_b_moment_estimate = self.B_moment_estimate[i][j] / (1 - beta_correction);
                        const bc_b_moment_estimate_2 = self.B_moment_estimate_2[i][j] / (1 - beta_2_correction);
                        const prev_b = model_ptr.layers[i].B[j];

                        model_ptr.layers[i].B[j] = prev_b - lr / (math.sqrt(bc_b_moment_estimate_2) + epsilon) * bc_b_moment_estimate;
                    }
                }

                self.t += 1;
            }
        },
    }
}
