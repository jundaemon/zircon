const std = @import("std");

const math = std.math;
const Random = std.Random;

fn uni_float(rand: Random) f32 {
    return rand.float(f32) * 2 - 1;
}

pub const Activation = enum { none, tanh };
pub const LossFn = enum { mse };
pub const Optimizer = enum { sgd };

fn Neuron(
    comptime in: usize,
    comptime activ: Activation,
    comptime optim: Optimizer,
    comptime lr: f32,
) type {
    return struct {
        const Self = @This();

        input: @Vector(in, f32) = @splat(0),
        weight: @Vector(in, f32),
        weight_grad: @Vector(in, f32) = @splat(0),
        bias: f32,
        bias_grad: f32 = 0,
        out: f32 = 0,

        fn init(rand: Random) Self {
            var weight: [in]f32 = undefined;
            for (0..in) |i| weight[i] = uni_float(rand);

            return .{ .weight = weight, .bias = uni_float(rand) };
        }

        fn forward(self: *Self, input: @Vector(in, f32)) f32 {
            self.input = input;

            const pre_activ = @reduce(.Add, self.weight * input) + self.bias;
            self.out = switch (activ) {
                .tanh => math.tanh(pre_activ),
                .none => pre_activ,
            };

            return self.out;
        }

        fn backward(self: *Self, out_grad: f32) @Vector(in, f32) {
            const pre_activ_grad = switch (activ) {
                .none => out_grad,
                .tanh => out_grad * (1 - math.pow(f32, self.out, 2)),
            };
            const vec_pre_activ_grad = @as(@Vector(in, f32), @splat(pre_activ_grad));

            self.weight_grad += self.input * vec_pre_activ_grad;
            self.bias_grad += pre_activ_grad;

            return self.weight * vec_pre_activ_grad;
        }

        fn step(self: *Self) void {
            switch (optim) {
                .sgd => {
                    self.weight -= @as(@Vector(in, f32), @splat(lr)) * self.weight_grad;
                    self.bias -= lr * self.bias_grad;
                },
            }
        }

        fn zero_grad(self: *Self) void {
            self.weight_grad = @as(@Vector(in, f32), @splat(0));
            self.bias_grad = 0;
        }
    };
}

fn Layer(
    comptime in: usize,
    comptime out: usize,
    comptime activ: Activation,
    comptime optim: Optimizer,
    comptime lr: f32,
) type {
    return struct {
        const Self = @This();

        neurons: [out]Neuron(in, activ, optim, lr),

        fn init(rand: Random) Self {
            var neurons: [out]Neuron(in, activ, optim, lr) = undefined;
            for (0..out) |i| neurons[i] = .init(rand);

            return .{ .neurons = neurons };
        }

        fn forward(self: *Self, input: @Vector(in, f32)) @Vector(out, f32) {
            var outs: [out]f32 = undefined;
            for (0..out) |i| outs[i] = self.neurons[i].forward(input);

            return outs;
        }

        fn backward(self: *Self, out_grad: @Vector(out, f32)) @Vector(in, f32) {
            var in_grads: @Vector(in, f32) = @splat(0);
            inline for (0..out) |i| in_grads += self.neurons[i].backward(out_grad[i]);

            return in_grads;
        }

        fn step(self: *Self) void {
            for (0..out) |i| self.neurons[i].step();
        }

        fn zero_grad(self: *Self) void {
            for (0..out) |i| self.neurons[i].zero_grad();
        }
    };
}

pub fn MLP(
    comptime in: usize,
    comptime n: usize,
    comptime outs: [n]usize,
    comptime activ: [n]Activation,
    comptime loss_fn: LossFn,
    comptime optim: Optimizer,
    comptime lr: f32,
    comptime seed: u64,
) type {
    const dims = [1]usize{in} ++ outs;

    var layer_types: [n]type = undefined;
    var out_types: [n]type = undefined;
    var in_types: [n]type = undefined;
    for (0..n) |i| {
        layer_types[i] = Layer(dims[i], dims[i + 1], activ[i], optim, lr);
        out_types[i] = @Vector(outs[i], f32);
        in_types[i] = @Vector(dims[i], f32);
    }

    const Layers = @Tuple(&layer_types);
    const Outs = @Tuple(&out_types);
    const Ins = @Tuple(&in_types);

    return struct {
        const Self = @This();

        layers: Layers,
        loss_grad: @Vector(dims[n], f32) = @splat(0),

        pub fn init() Self {
            var prng: Random.DefaultPrng = .init(seed);
            const rand = prng.random();

            var layers: Layers = undefined;
            inline for (0..n) |i| layers[i] = .init(rand);

            return .{ .layers = layers };
        }

        pub fn forward(self: *Self, input: @Vector(in, f32)) @Vector(dims[n], f32) {
            var inc_outs: Outs = undefined;
            inline for (0..n) |i| {
                if (i == 0) {
                    inc_outs[i] = self.layers[i].forward(input);
                } else {
                    inc_outs[i] = self.layers[i].forward(inc_outs[i - 1]);
                }
            }

            return inc_outs[n - 1];
        }

        pub fn loss(self: *Self, pred: @Vector(dims[n], f32), actual: @Vector(dims[n], f32)) f32 {
            switch (loss_fn) {
                .mse => {
                    var abs_err: @Vector(dims[n], f32) = pred - actual;
                    self.loss_grad = @as(@Vector(dims[n], f32), @splat(2)) * abs_err / @as(@Vector(dims[n], f32), @splat(dims[n]));
                    abs_err *= abs_err;
                    return @reduce(.Add, abs_err) / dims[n];
                },
            }
        }

        pub fn backward(self: *Self) void {
            var inc_in_grads: Ins = undefined;
            inline for (0..n) |i| {
                const rev_i = n - i - 1;
                if (rev_i == n - 1) {
                    inc_in_grads[rev_i] = self.layers[rev_i].backward(self.loss_grad);
                } else {
                    inc_in_grads[rev_i] = self.layers[rev_i].backward(inc_in_grads[rev_i + 1]);
                }
            }
        }

        pub fn step(self: *Self) void {
            inline for (0..n) |i| self.layers[i].step();
        }

        pub fn zero_grad(self: *Self) void {
            inline for (0..n) |i| self.layers[i].zero_grad();
        }
    };
}
