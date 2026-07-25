const std = @import("std");

const math = std.math;
const mem = std.mem;
const Random = std.Random;
const testing = std.testing;

fn uni_float(rand: Random) f32 {
    return rand.float(f32) * 2 - 1;
}

pub const Activation = enum { none, tanh };
pub const Optimizer = enum { sgd };

fn Neuron(
    comptime in: usize,
    comptime f: Activation,
    comptime optim: Optimizer,
    comptime lr: f32,
) type {
    return struct {
        const Self = @This();

        X: @Vector(in, f32) = @splat(0),
        W: @Vector(in, f32),
        do_dW: @Vector(in, f32) = @splat(0),
        b: f32,
        do_db: f32 = 0,
        a: f32 = 0,

        fn init(rand: Random) Self {
            var weights: [in]f32 = undefined;
            for (0..in) |i| weights[i] = uni_float(rand);

            return .{ .W = weights, .b = uni_float(rand) };
        }

        fn forward(self: *Self, X: @Vector(in, f32)) f32 {
            self.X = X;

            const z = @reduce(.Add, self.W * X) + self.b;
            self.a = switch (f) {
                .tanh => math.tanh(z),
                .none => z,
            };

            return self.a;
        }

        fn backward(self: *Self, do_da: f32) @Vector(in, f32) {
            const do_dz = switch (f) {
                .none => do_da,
                .tanh => do_da * (1 - math.pow(f32, self.a, 2)),
            };
            const vec_do_dz = @as(@Vector(in, f32), @splat(do_dz));

            self.do_dW += self.X * vec_do_dz;
            self.do_db += do_dz;

            return self.W * vec_do_dz;
        }

        fn step(self: *Self) void {
            switch (optim) {
                .sgd => {
                    self.W -= @as(@Vector(in, f32), @splat(lr)) * self.do_dW;
                    self.b -= lr * self.do_db;
                },
            }
        }

        fn zero_grad(self: *Self) void {
            self.do_dW = @as(@Vector(in, f32), @splat(0));
            self.do_db = 0;
        }
    };
}

fn Layer(
    comptime in: usize,
    comptime out: usize,
    comptime f: Activation,
    comptime optim: Optimizer,
    comptime lr: f32,
) type {
    return struct {
        const Self = @This();

        neurons: [out]Neuron(in, f, optim, lr),

        fn init(rand: Random) Self {
            var neurons: [out]Neuron(in, f, optim, lr) = undefined;
            for (0..out) |i| neurons[i] = .init(rand);

            return .{ .neurons = neurons };
        }

        fn forward(self: *Self, X: @Vector(in, f32)) @Vector(out, f32) {
            var A: [out]f32 = undefined;
            for (0..out) |i| A[i] = self.neurons[i].forward(X);

            return A;
        }

        fn backward(self: *Self, do_dA: @Vector(out, f32)) @Vector(in, f32) {
            var do_dX: @Vector(in, f32) = @splat(0);
            inline for (0..out) |i| do_dX += self.neurons[i].backward(do_dA[i]);

            return do_dX;
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
    comptime f: [n]Activation,
    comptime optim: Optimizer,
    comptime lr: f32,
    comptime seed: u64,
) type {
    const dims = [1]usize{in} ++ outs;

    var layer_types: [n]type = undefined;
    var out_types: [n]type = undefined;
    var in_types: [n]type = undefined;
    for (0..n) |i| {
        layer_types[i] = Layer(dims[i], dims[i + 1], f[i], optim, lr);
        out_types[i] = @Vector(outs[i], f32);
        in_types[i] = @Vector(dims[i], f32);
    }

    const Layers = @Tuple(&layer_types);
    const Outs = @Tuple(&out_types);
    const Ins = @Tuple(&in_types);

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

        pub fn forward(self: *Self, X: @Vector(in, f32)) @Vector(dims[n], f32) {
            var inc_A: Outs = undefined;
            inline for (0..n) |i| {
                if (i == 0) {
                    inc_A[i] = self.layers[i].forward(X);
                } else {
                    inc_A[i] = self.layers[i].forward(inc_A[i - 1]);
                }
            }

            return inc_A[n - 1];
        }

        pub fn backward(self: *Self, dL_dycap: @Vector(dims[n], f32)) void {
            var inc_dA: Ins = undefined;
            inline for (0..n) |i| {
                const rev_i = n - i - 1;
                if (rev_i == n - 1) {
                    inc_dA[rev_i] = self.layers[rev_i].backward(dL_dycap);
                } else {
                    inc_dA[rev_i] = self.layers[rev_i].backward(inc_dA[rev_i + 1]);
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

test "forward pass" {
    var model: MLP(2, 2, .{ 1, 1 }, .{ .tanh, .none }, .sgd, 0.0001, 1) = .init();
    const l1n1 = model.layers[0].neurons[0];
    const l2n1 = model.layers[1].neurons[0];

    const ycap1 = model.forward(.{ 0, 0 });
    try testing.expectEqual(math.tanh(l1n1.b) * l2n1.W[0] + l2n1.b, ycap1[0]);

    const ycap2 = model.forward(.{ 1, 0 });
    try testing.expectEqual(math.tanh(l1n1.W[0] + l1n1.b) * l2n1.W[0] + l2n1.b, ycap2[0]);

    const ycap3 = model.forward(.{ 0, 1 });
    try testing.expectEqual(math.tanh(l1n1.W[1] + l1n1.b) * l2n1.W[0] + l2n1.b, ycap3[0]);

    const ycap4 = model.forward(.{ 1, 1 });
    try testing.expectEqual(math.tanh(l1n1.W[0] + l1n1.W[1] + l1n1.b) * l2n1.W[0] + l2n1.b, ycap4[0]);
}

test "backprop + sgd + zero grad" {
    const lr = 0.0001;
    var model: MLP(1, 3, .{ 1, 2, 1 }, .{ .none, .tanh, .none }, .sgd, lr, 1) = .init();

    _ = model.forward(.{1});
    model.backward(.{1});
    const l1n1 = model.layers[0].neurons[0];
    const l2n1 = model.layers[1].neurons[0];
    const l2n2 = model.layers[1].neurons[1];
    const l3n1 = model.layers[2].neurons[0];
    const l2n1_do_dz = l3n1.W[0] * (1 - math.pow(f32, l2n1.a, 2));
    const l2n2_do_dz = l3n1.W[1] * (1 - math.pow(f32, l2n2.a, 2));
    try testing.expectEqual(l3n1.X, l3n1.do_dW);
    try testing.expectEqual(1, l3n1.do_db);
    try testing.expectEqual(l2n1_do_dz * l2n1.X[0], l2n1.do_dW[0]);
    try testing.expectEqual(l2n1_do_dz, l2n1.do_db);
    try testing.expectEqual(l2n2_do_dz * l2n2.X[0], l2n2.do_dW[0]);
    try testing.expectEqual(l2n2_do_dz, l2n2.do_db);
    try testing.expectEqual((l2n1_do_dz * l2n1.W[0] + l2n2_do_dz * l2n2.W[0]) * l1n1.X[0], l1n1.do_dW[0]);
    try testing.expectEqual((l2n1_do_dz * l2n1.W[0] + l2n2_do_dz * l2n2.W[0]), l1n1.do_db);

    model.step();
    const l1n1_p = model.layers[0].neurons[0];
    const l2n1_p = model.layers[1].neurons[0];
    const l2n2_p = model.layers[1].neurons[1];
    const l3n1_p = model.layers[2].neurons[0];
    try testing.expectEqual(l1n1.W[0] - lr * l1n1.do_dW[0], l1n1_p.W[0]);
    try testing.expectEqual(l1n1.b - lr * l1n1.do_db, l1n1_p.b);
    try testing.expectEqual(l2n1.W[0] - lr * l2n1.do_dW[0], l2n1_p.W[0]);
    try testing.expectEqual(l2n1.b - lr * l2n1.do_db, l2n1_p.b);
    try testing.expectEqual(l2n2.W[0] - lr * l2n2.do_dW[0], l2n2_p.W[0]);
    try testing.expectEqual(l2n2.b - lr * l2n2.do_db, l2n2_p.b);
    try testing.expectEqual(l3n1.W - @as(@Vector(2, f32), @splat(lr)) * l3n1.do_dW, l3n1_p.W);
    try testing.expectEqual(l3n1.b - lr * l3n1.do_db, l3n1_p.b);

    model.zero_grad();
    const l1n1_pp = model.layers[0].neurons[0];
    const l2n1_pp = model.layers[1].neurons[0];
    const l2n2_pp = model.layers[1].neurons[1];
    const l3n1_pp = model.layers[2].neurons[0];
    try testing.expectEqual(0, l1n1_pp.do_dW[0]);
    try testing.expectEqual(0, l1n1_pp.do_db);
    try testing.expectEqual(0, l2n1_pp.do_dW[0]);
    try testing.expectEqual(0, l2n1_pp.do_db);
    try testing.expectEqual(0, l2n2_pp.do_dW[0]);
    try testing.expectEqual(0, l2n2_pp.do_db);
    try testing.expectEqual(@as(@Vector(2, f32), @splat(0)), l3n1_pp.do_dW);
    try testing.expectEqual(0, l3n1_pp.do_db);
}
