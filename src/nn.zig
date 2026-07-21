const std = @import("std");

const math = std.math;
const Random = std.Random;
const testing = std.testing;

fn uni_float(rand: Random) f32 {
    return rand.float(f32) * 2 - 1;
}

pub const Activation = enum { none, tanh };

fn Neuron(comptime in: usize, comptime act: Activation) type {
    return struct {
        const Self = @This();

        weights: @Vector(in, f32),
        bias: f32,
        act: Activation,

        fn init(rand: Random) Self {
            var weights: [in]f32 = undefined;
            for (0..in) |i| weights[i] = uni_float(rand);

            return .{ .weights = weights, .bias = uni_float(rand), .act = act };
        }

        fn forward(self: Self, input: @Vector(in, f32)) f32 {
            const z = @reduce(.Add, self.weights * input) + self.bias;

            return switch (act) {
                .tanh => math.tanh(z),
                .none => z,
            };
        }
    };
}

fn Layer(comptime in: usize, comptime out: usize, comptime act: Activation) type {
    return struct {
        const Self = @This();

        neurons: [out]Neuron(in, act),

        fn init(rand: Random) Self {
            var neurons: [out]Neuron(in, act) = undefined;
            for (0..out) |i| neurons[i] = .init(rand);

            return .{ .neurons = neurons };
        }

        fn forward(self: Self, input: @Vector(in, f32)) @Vector(out, f32) {
            var outs: [out]f32 = undefined;
            for (0..out, self.neurons) |i, neuron| outs[i] = neuron.forward(input);

            return outs;
        }
    };
}

pub fn Sequential(comptime in: usize, comptime n: usize, comptime outs: [n]usize, comptime acts: [n]Activation, comptime seed: u64) type {
    const dims = [1]usize{in} ++ outs;

    var layer_types: [n]type = undefined;
    for (0..n) |i| layer_types[i] = Layer(dims[i], dims[i + 1], acts[i]);
    const Layers = @Tuple(&layer_types);

    var out_types: [n]type = undefined;
    for (0..n, outs) |i, out| out_types[i] = @Vector(out, f32);
    const Outs = @Tuple(&out_types);

    return struct {
        const Self = @This();

        layers: Layers,

        pub fn init() Self {
            var prng: Random.DefaultPrng = .init(seed);
            const rand = prng.random();

            var layers: Layers = undefined;
            inline for (0..n) |i| layers[i] = .init(rand);

            return .{ .layers = layers };
        }

        pub fn forward(self: Self, input: @Vector(in, f32)) @Vector(dims[n], f32) {
            var inc_outs: Outs = undefined;
            inline for (0..n, self.layers) |i, layer| {
                if (i == 0) {
                    inc_outs[i] = layer.forward(input);
                } else {
                    inc_outs[i] = layer.forward(inc_outs[i - 1]);
                }
            }

            return inc_outs[n - 1];
        }
    };
}

test "forward pass" {
    const model_1 = Sequential(2, 2, .{ 1, 1 }, .{ .tanh, .none }, 1).init();
    const l1n1_1 = model_1.layers[0].neurons[0];
    const l2n1_1 = model_1.layers[1].neurons[0];

    const model_2 = Sequential(2, 2, .{ 1, 1 }, .{ .none, .tanh }, 1).init();
    const l1n1_2 = model_2.layers[0].neurons[0];
    const l2n1_2 = model_2.layers[1].neurons[0];

    const out_1 = model_1.forward(.{ 0, 0 });
    try testing.expectEqual(math.tanh(l1n1_1.bias) * l2n1_1.weights[0] + l2n1_1.bias, out_1[0]);

    const out_2 = model_2.forward(.{ 0, 0 });
    try testing.expectEqual(math.tanh(l1n1_2.bias * l2n1_2.weights[0] + l2n1_2.bias), out_2[0]);

    const out_3 = model_1.forward(.{ 1, 0 });
    try testing.expectEqual(math.tanh(l1n1_1.weights[0] + l1n1_1.bias) * l2n1_1.weights[0] + l2n1_1.bias, out_3[0]);

    const out_4 = model_2.forward(.{ 1, 0 });
    try testing.expectEqual(math.tanh((l1n1_2.weights[0] + l1n1_2.bias) * l2n1_2.weights[0] + l2n1_2.bias), out_4[0]);

    const out_5 = model_1.forward(.{ 0, 1 });
    try testing.expectEqual(math.tanh(l1n1_1.weights[1] + l1n1_1.bias) * l2n1_1.weights[0] + l2n1_1.bias, out_5[0]);

    const out_6 = model_2.forward(.{ 0, 1 });
    try testing.expectEqual(math.tanh((l1n1_2.weights[1] + l1n1_2.bias) * l2n1_2.weights[0] + l2n1_2.bias), out_6[0]);

    const out_7 = model_1.forward(.{ 1, 1 });
    try testing.expectEqual(math.tanh(l1n1_1.weights[0] + l1n1_1.weights[1] + l1n1_1.bias) * l2n1_1.weights[0] + l2n1_1.bias, out_7[0]);

    const out_8 = model_2.forward(.{ 1, 1 });
    try testing.expectEqual(math.tanh((l1n1_2.weights[0] + l1n1_2.weights[1] + l1n1_2.bias) * l2n1_2.weights[0] + l2n1_2.bias), out_8[0]);
}
