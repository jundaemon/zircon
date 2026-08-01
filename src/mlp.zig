const std = @import("std");
const math = std.math;
const mem = std.mem;
const fmt = std.fmt;
const testing = std.testing;
const Random = std.Random;
const Io = std.Io;

const init = @import("init");
const uniform_float = init.uniform_float;

pub const Activation = enum { none, tanh };
pub const LossFunc = enum { mse };
pub const Optimizer = enum { sgd };

fn Layer(
    comptime in: usize,
    comptime out: usize,
    comptime activ: Activation,
    comptime optim: Optimizer,
    comptime lr: f32,
) type {
    return struct {
        weights: [out][in]f32,
        weights_grad: [out][in]f32 = [_][in]f32{@splat(0)} ** out,
        biases: [out]f32,
        biases_grad: [out]f32 = @splat(0),
        outs: [out]f32 = @splat(0),
        input: [in]f32 = @splat(0),

        const Self = @This();
        fn init(rand: Random) Self {
            var weights: [out][in]f32 = undefined;
            var biases: [out]f32 = undefined;
            for (0..out) |i| {
                var weight: [in]f32 = undefined;
                for (0..in) |j| weight[j] = uniform_float(rand);

                weights[i] = weight;
                biases[i] = uniform_float(rand);
            }

            return .{ .weights = weights, .biases = biases };
        }

        fn load(weights: [out][in]f32, biases: [out]f32) Self {
            return .{ .weights = weights, .biases = biases };
        }

        fn forward(self: *Self, input: [in]f32) [out]f32 {
            self.input = input;

            for (0..out) |i| {
                const pre_activ = @reduce(.Add, @as(@Vector(in, f32), input) * self.weights[i]) + self.biases[i];
                self.outs[i] = switch (activ) {
                    .none => pre_activ,
                    .tanh => math.tanh(pre_activ),
                };
            }

            return self.outs;
        }

        fn backward(self: *Self, out_grad: [out]f32) [in]f32 {
            var vec_in_grad: @Vector(in, f32) = @splat(0);
            for (0..out) |i| {
                const pre_activ_grad = switch (activ) {
                    .none => out_grad[i],
                    .tanh => out_grad[i] * (1 - math.pow(f32, self.outs[i], 2)),
                };
                const vec_pre_activ_grad: @Vector(in, f32) = @splat(pre_activ_grad);
                self.weights_grad[i] = self.weights_grad[i] + self.input * vec_pre_activ_grad;
                self.biases_grad[i] += pre_activ_grad;

                vec_in_grad += self.weights[i] * vec_pre_activ_grad;
            }

            return vec_in_grad;
        }

        fn step(self: *Self) void {
            switch (optim) {
                .sgd => {
                    for (0..out) |i| self.weights[i] = self.weights[i] - @as(@Vector(in, f32), @splat(lr)) * self.weights_grad[i];
                    self.biases = self.biases - @as(@Vector(out, f32), @splat(lr)) * self.biases_grad;
                },
            }
        }

        fn zero_grad(self: *Self) void {
            self.weights_grad = [_][in]f32{@splat(0)} ** out;
            self.biases_grad = @splat(0);
        }
    };
}

pub fn MLP(
    comptime in: usize,
    comptime n: usize,
    comptime outs: [n]usize,
    comptime activ: [n]Activation,
    comptime loss_func: LossFunc,
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
        out_types[i] = [outs[i]]f32;
        in_types[i] = [dims[i]]f32;
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

        pub fn load(io: Io, path: []const u8) !Self {
            var file: Io.File = try Io.Dir.cwd().openFile(io, path, .{});
            defer file.close(io);

            var buf: [1_024]u8 = undefined;
            var reader: Io.File.Reader = file.reader(io, &buf);
            const interface = &reader.interface;

            var layers: Layers = undefined;
            inline for (0..n) |i| {
                const out = dims[i + 1];

                var weight_bytes: [out * dims[i] * 4]u8 = undefined;
                try interface.readSliceAll(&weight_bytes);

                var bias_bytes: [out * 4]u8 = undefined;
                try interface.readSliceAll(&bias_bytes);

                layers[i] = .load(
                    mem.bytesToValue([out][dims[i]]f32, &weight_bytes),
                    mem.bytesToValue([out]f32, &bias_bytes),
                );
            }

            return .{ .layers = layers };
        }

        pub fn save(self: Self, io: Io, path: []const u8) !void {
            var file: Io.File = try Io.Dir.cwd().createFile(io, path, .{});
            defer file.close(io);

            var buf: [1_024]u8 = undefined;
            var writer: Io.File.Writer = file.writer(io, &buf);
            const interface = &writer.interface;
            inline for (0..n) |i| {
                const out = dims[i + 1];

                var weight: [out][dims[i]]f32 = undefined;
                var bias: [out]f32 = undefined;
                for (0..out) |j| {
                    weight[j] = self.layers[i].weights[j];
                    bias[j] = self.layers[i].biases[j];
                }

                try interface.writeAll(mem.asBytes(&weight));
                try interface.writeAll(mem.asBytes(&bias));
            }

            try interface.flush();
        }

        pub fn forward(self: *Self, input: [in]f32) [dims[n]]f32 {
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

        pub fn loss(self: Self, predicted: [dims[n]]f32, expected: [dims[n]]f32) struct { f32, [dims[n]]f32 } {
            _ = self;
            switch (loss_func) {
                .mse => {
                    const ae = @as(@Vector(dims[n], f32), predicted) - expected;
                    return .{
                        @reduce(.Add, ae * ae) / dims[n],
                        ae * @as(@Vector(dims[n], f32), @splat(2 / dims[n])),
                    };
                },
            }
        }

        pub fn backward(self: *Self, loss_grad: [dims[n]]f32) void {
            var inc_in_grads: Ins = undefined;
            inline for (0..n) |i| {
                const rev_i = n - i - 1;
                if (rev_i == n - 1) {
                    inc_in_grads[rev_i] = self.layers[rev_i].backward(loss_grad);
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

test "mlp 1" {
    const io = testing.io;
    var file: Io.File = try Io.Dir.cwd().createFile(io, "tests/cases/mlp_1_loss.txt", .{});
    defer file.close(io);

    var model: MLP(3, 3, .{ 10, 15, 3 }, .{ .tanh, .tanh, .none }, .mse, .sgd, 0.0001, 1) = .init();
    try model.save(io, "tests/cases/mlp_1_init.bin");

    const pred = model.forward(.{ 1, 2, 3 });
    const loss, const loss_grad = model.loss(pred, .{ 0.1, 0.2, 0.3 });
    var buf: [10]u8 = undefined;
    const loss_str = try fmt.bufPrint(&buf, "{d}", .{loss});
    try file.writeStreamingAll(io, loss_str);

    model.backward(loss_grad);
    model.step();
    try model.save(io, "tests/cases/mlp_1_final.bin");
}

test "mlp 2" {
    const io = testing.io;
    var file: Io.File = try Io.Dir.cwd().createFile(io, "tests/cases/mlp_2_loss.txt", .{});
    defer file.close(io);

    var model: MLP(4, 4, .{ 7, 10, 11, 5 }, .{ .tanh, .tanh, .tanh, .none }, .mse, .sgd, 0.0001, 1) = .init();
    try model.save(io, "tests/cases/mlp_2_init.bin");

    const pred_1 = model.forward(.{ 1, 2, 3, 4 });
    const loss_1, const loss_grad_1 = model.loss(pred_1, .{ 0.1, 0.2, 0.3, 0.4, 0.5 });
    var buf_1: [10]u8 = undefined;
    const loss_1_str = try fmt.bufPrint(&buf_1, "{d}\n", .{loss_1});
    try file.writeStreamingAll(io, loss_1_str);

    model.backward(loss_grad_1);
    model.step();
    try model.save(io, "tests/cases/mlp_2_updated.bin");

    model.zero_grad();
    const pred_2 = model.forward(.{ 1, 2, 3, 4 });
    const loss_2, const loss_grad_2 = model.loss(pred_2, .{ 0.1, 0.2, 0.3, 0.4, 0.5 });
    var buf_2: [10]u8 = undefined;
    const loss_2_str = try fmt.bufPrint(&buf_2, "{d}\n", .{loss_2});
    try file.writeStreamingAll(io, loss_2_str);

    model.backward(loss_grad_2);
    model.step();
    try model.save(io, "tests/cases/mlp_2_final.bin");
}
