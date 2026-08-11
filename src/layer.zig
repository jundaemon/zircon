const std = @import("std");
const math = std.math;
const Random = std.Random;

const function = @import("function");
const Activation = function.Activation;

fn custom_rand_f32(rand: Random, bound: f32) f32 {
    return rand.float(f32) * bound * 2 - bound;
}

pub fn Layer(comptime in: usize, comptime out: usize, comptime activation: Activation) type {
    return struct {
        weights: [out][in]f32,
        weight_grads: [out][in]f32 = [_][in]f32{@splat(0)} ** out,
        biases: [out]f32,
        bias_grads: [out]f32 = @splat(0),
        outs: [out]f32 = @splat(0),
        input: [in]f32 = @splat(0),

        const Self = @This();
        pub fn init(rand: Random) Self {
            const weight_bound = switch (activation) {
                .None, .Tanh => math.sqrt(6 / @as(f32, in + out)),
                .ReLU => math.sqrt(6 / @as(f32, in)),
            };
            const bias_bound = 1 / math.sqrt(@as(f32, in));

            var layer_weights: [out][in]f32 = undefined;
            var layer_biases: [out]f32 = undefined;
            for (0..out) |i| {
                var neuron_weights: [in]f32 = undefined;
                for (0..in) |j| neuron_weights[j] = custom_rand_f32(rand, weight_bound);

                const neuron_bias = custom_rand_f32(rand, bias_bound);

                layer_weights[i] = neuron_weights;
                layer_biases[i] = neuron_bias;
            }

            return .{ .weights = layer_weights, .biases = layer_biases };
        }

        pub fn load(weights: [out][in]f32, biases: [out]f32) Self {
            return .{ .weights = weights, .biases = biases };
        }

        pub fn forward(self: *Self, input: [in]f32) [out]f32 {
            self.input = input;
            self.outs = @splat(0);

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

        fn accumulate_grads(self: *Self, i: usize, out_grad: f32, in_grads: []f32) void {
            for (0..in) |j| {
                self.weight_grads[i][j] += self.input[j] * out_grad;
                in_grads[j] += self.weights[i][j] * out_grad;
            }
            self.bias_grads[i] += out_grad;
        }

        pub fn backward(self: *Self, out_grads: [out]f32) [in]f32 {
            var in_grads: [in]f32 = @splat(0);
            switch (activation) {
                .None => {
                    for (0..out) |i| self.accumulate_grads(i, out_grads[i], &in_grads);
                },
                .Tanh => {
                    for (0..out) |i| {
                        const out_grad = out_grads[i] * (1 - math.pow(f32, self.outs[i], 2));
                        self.accumulate_grads(i, out_grad, &in_grads);
                    }
                },
                .ReLU => {
                    for (0..out) |i| {
                        const out_grad = out_grads[i] * if (self.outs[i] > 0) @as(f32, 1) else 0;
                        self.accumulate_grads(i, out_grad, &in_grads);
                    }
                },
            }

            return in_grads;
        }
    };
}
