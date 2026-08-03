const std = @import("std");
const math = std.math;
const mem = std.mem;
const fmt = std.fmt;
const testing = std.testing;
const Random = std.Random;
const Io = std.Io;

const lossf = @import("lossf");

pub const Activation = enum { None, Tanh, ReLU };
pub const Optimizer = enum { SGD };

fn randFloat(rand: Random, bound: f32) f32 {
    return rand.float(f32) * bound * 2 - bound;
}

/// private comptime struct that holds all information needed for forward and backward passes
/// "in" is the size of input to the layer
/// "out" is the number of neurons within the layer
/// "activ" is the activation function for the the layer
/// "optim" is the optimizer for the network
/// "lr" is the learning rate for the network
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

        /// initializes the weights and biases according to the activation function of the layer
        /// if the layer uses a symmetric activation function, weights are initialized using Xavier uniform distribution
        /// if the layer uses an asymmetric activation function, weights are initialized using Kaiming uniform distribution in fan-in mode
        /// biases are initialized using LeCun uniform distribution regardless
        fn init(rand: Random) Self {
            const w_bound = switch (activ) {
                .None, .Tanh => math.sqrt(6 / @as(f32, in + out)),
                .ReLU => math.sqrt(6 / @as(f32, in)),
            };
            const b_bound = 1 / math.sqrt(@as(f32, in));

            var weights: [out][in]f32 = undefined;
            var biases: [out]f32 = undefined;
            for (0..out) |i| {
                var weight: [in]f32 = undefined;
                for (0..in) |j| weight[j] = randFloat(rand, w_bound);

                weights[i] = weight;
                biases[i] = randFloat(rand, b_bound);
            }

            return .{ .weights = weights, .biases = biases };
        }

        fn load(weights: [out][in]f32, biases: [out]f32) Self {
            return .{ .weights = weights, .biases = biases };
        }

        /// performs a forward pass using a = f(z), z = wx + b, where f is the activation function if any
        /// the output is returned and will be passed on to the next layer
        /// the input and outputs are saved for use in backpropagation after
        fn forward(self: *Self, input: [in]f32) [out]f32 {
            self.input = input;

            for (0..out) |i| {
                const pre_activ = @reduce(.Add, @as(@Vector(in, f32), input) * self.weights[i]) + self.biases[i];
                self.outs[i] = switch (activ) {
                    .None => pre_activ,
                    .Tanh => math.tanh(pre_activ),
                    .ReLU => if (pre_activ > 0) pre_activ else 0,
                };
            }

            return self.outs;
        }

        /// performs backpropagation on the layer and returns a vector of gradients
        /// this vector is passed on to the previous layer
        fn backward(self: *Self, out_grad: [out]f32) [in]f32 {
            var vec_in_grad: @Vector(in, f32) = @splat(0);
            for (0..out) |i| {
                const pre_activ_grad = switch (activ) {
                    .None => out_grad[i],
                    .Tanh => out_grad[i] * (1 - math.pow(f32, self.outs[i], 2)),
                    .ReLU => out_grad[i] * if (self.outs[i] > 0) @as(f32, 1) else 0,
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
                .SGD => {
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

/// public comptime struct that serves as a consolidation of all layers
/// "in" is the size of input to the network
/// "n" is the number of layers
/// "outs" is the number of neurons in each layer
/// "activ" is the activation function used for each layer
/// "optim" is the optimizer for the network
/// "lr" is the learning rate for the network
/// "seed" allows for replicable initialization of weights and biases
pub fn MLP(
    comptime in: usize,
    comptime n: usize,
    comptime outs: [n]usize,
    comptime activ: [n]Activation,
    comptime optim: Optimizer,
    comptime lr: f32,
    comptime seed: u64,
) type {
    if (n < 1) @compileError("model needs at least 1 layer");
    if (in < 1) @compileError("input should be of size 1 or larger");

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
        layers: Layers,

        const Self = @This();

        pub fn init() Self {
            var prng: Random.DefaultPrng = .init(seed);
            const rand = prng.random();

            var layers: Layers = undefined;
            inline for (0..n) |i| layers[i] = .init(rand);

            return .{ .layers = layers };
        }

        /// reads a given file for the weights and biases for the network incrementally
        /// based on how it was saved in the "save" method, reading any other file that was not
        /// created using this library will likely raise an error or lead to garbage weights
        pub fn load(io: Io, path: []const u8) !Self {
            var file: Io.File = try Io.Dir.cwd().openFile(io, path, .{});
            defer file.close(io);

            var buf: [4_096]u8 = undefined;
            var reader: Io.File.Reader = file.reader(io, &buf);
            const interface = &reader.interface;

            var layers: Layers = undefined;
            inline for (0..n) |i| {
                const in_ = dims[i];
                const out = dims[i + 1];

                var weight_bytes: [out * in_ * 4]u8 = undefined;
                try interface.readSliceAll(&weight_bytes);

                var bias_bytes: [out * 4]u8 = undefined;
                try interface.readSliceAll(&bias_bytes);

                layers[i] = .load(
                    mem.bytesToValue([out][in_]f32, &weight_bytes),
                    mem.bytesToValue([out]f32, &bias_bytes),
                );
            }

            return .{ .layers = layers };
        }

        /// writes the weights and biases of each layer to the given path as bytes
        pub fn save(self: Self, io: Io, path: []const u8) !void {
            var file: Io.File = try Io.Dir.cwd().createFile(io, path, .{});
            defer file.close(io);

            var buf: [4_096]u8 = undefined;
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

        /// calculates the forward pass through the network
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

        /// performs backpropagation through the network
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

test "mlp" {
    const io = testing.io;
    var file: Io.File = try Io.Dir.cwd().createFile(io, "tests/cases/mlp_loss.txt", .{});
    defer file.close(io);

    var model: MLP(2, 5, .{ 3, 5, 7, 4, 1 }, .{ .Tanh, .ReLU, .Tanh, .ReLU, .None }, .SGD, 0.0001, 1) = .init();
    try model.save(io, "tests/cases/mlp_init_weights.bin");

    const pred = model.forward(.{ 1, 2 });
    const loss, const loss_grad = lossf.MSE(1, pred, .{0.1});

    var buf: [30]u8 = undefined;
    const loss_str = try fmt.bufPrint(&buf, "{d}\n", .{loss});
    try file.writeStreamingAll(io, loss_str);

    model.backward(loss_grad);
    model.step();
    model.zero_grad();
    try model.save(io, "tests/cases/mlp_updated_weights.bin");

    const pred_prime = model.forward(.{ 3, 2 });
    const loss_prime, const loss_grad_prime = lossf.MSE(1, pred_prime, .{0.2});

    var buf_prime: [30]u8 = undefined;
    const loss_prime_str = try fmt.bufPrint(&buf_prime, "{d}\n", .{loss_prime});
    try file.writeStreamingAll(io, loss_prime_str);

    model.backward(loss_grad_prime);
    model.step();
    try model.save(io, "tests/cases/mlp_final_weights.bin");
}
