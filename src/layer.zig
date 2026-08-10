const std = @import("std");
const math = std.math;
const Random = std.Random;

const functions = @import("functions");
const Activation = functions.Activation;

fn custom_rand_f32(rand: Random, bound: f32) f32 {
    return rand.float(f32) * bound * 2 - bound;
}

pub fn Layer(comptime in: usize, comptime out: usize, comptime activation: Activation) type {
    return struct {
        weights: [out][in]f32,
        weights_grad: [out][in]f32 = [_][in]f32{@splat(0)} ** out,
        biases: [out]f32,
        biases_grad: [out]f32 = @splat(0),
        outs: [out]f32 = @splat(0),
        input: [in]f32 = @splat(0),

        const Self = @This();
        pub fn init(rand: Random) Self {
            const w_bound = switch (activation) {
                .None, .Tanh => math.sqrt(6 / @as(f32, in + out)), // bounds for Xavier uniform initialization
                .ReLU => math.sqrt(6 / @as(f32, in)), // bounds for Kaiming uniform initialization in fan in mode
            };
            const b_bound = 1 / math.sqrt(@as(f32, in));

            var weights: [out][in]f32 = undefined;
            var biases: [out]f32 = undefined;
            for (0..out) |i| {
                var weight: [in]f32 = undefined;
                for (0..in) |j| weight[j] = custom_rand_f32(rand, w_bound);

                weights[i] = weight;
                biases[i] = custom_rand_f32(rand, b_bound);
            }

            return .{ .weights = weights, .biases = biases };
        }

        pub fn load(weights: [out][in]f32, biases: [out]f32) Self {
            return .{ .weights = weights, .biases = biases };
        }

        pub fn forward(self: *Self, input: [in]f32) [out]f32 {
            self.input = input; // inputs and outputs are saved for backpropagation

            self.outs = @splat(0); // clear outs from previous forward pass
            for (0..out) |i| {
                for (0..in) |j| self.outs[i] += input[j] * self.weights[i][j];
                self.outs[i] += self.biases[i];
            }
            switch (activation) {
                .None => {},
                .Tanh => {
                    for (0..out) |i| {
                        const pre_activation = self.outs[i];
                        self.outs[i] = math.tanh(pre_activation);
                    }
                },
                .ReLU => {
                    for (0..out) |i| {
                        const pre_activation = self.outs[i];
                        self.outs[i] = if (pre_activation > 0) pre_activation else 0;
                    }
                },
            }

            return self.outs;
        }

        pub fn accumulate_grads(self: *Self, i: usize, grad: f32, in_grad: []f32) void {
            for (0..in) |j| {
                self.weights_grad[i][j] += self.input[j] * grad;
                in_grad[j] += self.weights[i][j] * grad;
            }
            self.biases_grad[i] += grad;
        }

        pub fn backward(self: *Self, out_grad: [out]f32) [in]f32 {
            var in_grad: [in]f32 = @splat(0);
            switch (activation) {
                .None => {
                    for (0..out) |i| {
                        const grad = out_grad[i];
                        self.accumulate_grads(i, grad, &in_grad);
                    }
                },
                .Tanh => {
                    for (0..out) |i| {
                        const grad = out_grad[i] * (1 - math.pow(f32, self.outs[i], 2));
                        self.accumulate_grads(i, grad, &in_grad);
                    }
                },
                .ReLU => {
                    for (0..out) |i| {
                        const grad = out_grad[i] * if (self.outs[i] > 0) @as(f32, 1) else 0;
                        self.accumulate_grads(i, grad, &in_grad);
                    }
                },
            }

            return in_grad;
        }

        pub fn zero_grad(self: *Self) void {
            self.weights_grad = [_][in]f32{@splat(0)} ** out;
            self.biases_grad = @splat(0);
        }
    };
}
