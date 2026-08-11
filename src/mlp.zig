const std = @import("std");
const fmt = std.fmt;
const Io = std.Io;
const mem = std.mem;
const Random = std.Random;
const testing = std.testing;

const function = @import("function");
const Activation = function.Activation;

const layer = @import("layer");
const Layer = layer.Layer;

const optim = @import("optim");
const Optimizer = optim.Optimizer;

pub const MLPConfig = struct {
    in: usize,
    outs: []const usize,
    activations: []const Activation,
    seed: u64,

    pub fn check(comptime self: MLPConfig) void {
        if (self.in == 0) @compileError("in should be 1 or more");
        if (self.outs.len == 0 or self.activations.len == 0) @compileError("number of layers should be 1 or more");
        if (self.outs.len != self.activations.len) @compileError("outs and activations should have the same length");
        for (self.outs) |out| if (out == 0) @compileError("number of neurons in each layer should be 1 or more");
    }
};

pub fn MLP(comptime mlp_config: MLPConfig) type {
    mlp_config.check();
    const num_layers = mlp_config.outs.len;
    const dimensions = [1]usize{mlp_config.in} ++ mlp_config.outs;

    var layer_types: [num_layers]type = undefined;
    var in_types: [num_layers]type = undefined;
    var out_types: [num_layers]type = undefined;
    for (0..num_layers) |i| {
        const in = dimensions[i];
        const out = dimensions[i + 1];
        const activation = mlp_config.activations[i];

        layer_types[i] = Layer(in, out, activation);
        in_types[i] = [in]f32;
        out_types[i] = [out]f32;
    }

    const Layers = @Tuple(&layer_types);
    const Ins = @Tuple(&in_types);
    const Outs = @Tuple(&out_types);

    return struct {
        layers: Layers,

        const Self = @This();
        pub fn init() Self {
            var prng: Random.DefaultPrng = .init(mlp_config.seed);
            const rand = prng.random();

            var layers: Layers = undefined;
            inline for (0..num_layers) |i| layers[i] = .init(rand);

            return .{ .layers = layers };
        }

        pub fn load(io: Io, path: []const u8) !Self {
            var file: Io.File = try Io.Dir.cwd().openFile(io, path, .{});
            defer file.close(io);

            var buf: [4_096]u8 = undefined;
            var reader: Io.File.Reader = file.reader(io, &buf);
            const interface = &reader.interface;

            var layers: Layers = undefined;
            inline for (0..num_layers) |i| {
                const in = dimensions[i];
                const out = dimensions[i + 1];

                var layer_weights_bytes: [out * in * 4]u8 = undefined;
                try interface.readSliceAll(&layer_weights_bytes);

                var layer_biases_bytes: [out * 4]u8 = undefined;
                try interface.readSliceAll(&layer_biases_bytes);

                layers[i] = .load(
                    mem.bytesToValue([out][in]f32, &layer_weights_bytes),
                    mem.bytesToValue([out]f32, &layer_biases_bytes),
                );
            }

            return .{ .layers = layers };
        }

        pub fn save(self: Self, io: Io, path: []const u8) !void {
            var file: Io.File = try Io.Dir.cwd().createFile(io, path, .{});
            defer file.close(io);

            var buf: [4_096]u8 = undefined;
            var writer: Io.File.Writer = file.writer(io, &buf);
            const interface = &writer.interface;

            inline for (0..num_layers) |i| {
                const in = dimensions[i];
                const out = dimensions[i + 1];

                var layer_weights: [out][in]f32 = undefined;
                var layer_biases: [out]f32 = undefined;
                for (0..out) |j| {
                    const neuron_weights = self.layers[i].weights[j];
                    const neuron_bias = self.layers[i].biases[j];

                    layer_weights[j] = neuron_weights;
                    layer_biases[j] = neuron_bias;
                }

                try interface.writeAll(mem.asBytes(&layer_weights));
                try interface.writeAll(mem.asBytes(&layer_biases));
            }

            try interface.flush();
        }

        pub fn forward(self: *Self, input: [mlp_config.in]f32) [dimensions[num_layers]]f32 {
            var incremental_outs: Outs = undefined;
            inline for (0..num_layers) |i| {
                if (i == 0) {
                    incremental_outs[i] = self.layers[i].forward(input);
                } else {
                    incremental_outs[i] = self.layers[i].forward(incremental_outs[i - 1]);
                }
            }

            return incremental_outs[num_layers - 1];
        }

        pub fn backward(self: *Self, loss_grad: [dimensions[num_layers]]f32) void {
            var incremental_in_grads: Ins = undefined;
            inline for (0..num_layers) |i| {
                const reverse_i = num_layers - i - 1;
                if (reverse_i == num_layers - 1) {
                    incremental_in_grads[reverse_i] = self.layers[reverse_i].backward(loss_grad);
                } else {
                    incremental_in_grads[reverse_i] = self.layers[reverse_i].backward(incremental_in_grads[reverse_i + 1]);
                }
            }
        }
    };
}

test "mlp 1" {
    const io = testing.io;
    var file: Io.File = try Io.Dir.cwd().createFile(io, "tests/cases/mlp_1_losses", .{});
    defer file.close(io);

    const mlp_config = MLPConfig{
        .in = 2,
        .outs = &.{ 3, 5, 7, 4, 1 },
        .activations = &.{ .Tanh, .Tanh, .Tanh, .Tanh, .None },
        .seed = 1,
    };
    var model: MLP(mlp_config) = .init();
    try model.save(io, "tests/cases/mlp_1_initial_weights");

    const loss_fn = function.MSE;
    const loss_grad_fn = function.MSE_grad;
    var optimizer: Optimizer(.{
        .mlp_config = mlp_config,
        .optimizer = .SGD,
    }) = try .init(&model, 0.0001, null);

    const pred = model.forward(.{ 1, 2 });
    const loss = loss_fn(1, pred, .{0.1});
    const loss_grad = loss_grad_fn(1, pred, .{0.1});
    var buf: [30]u8 = undefined;
    const loss_str = try fmt.bufPrint(&buf, "{d}\n", .{loss});
    try file.writeStreamingAll(io, loss_str);

    model.backward(loss_grad);
    optimizer.step();
    optimizer.zero_grad();
    try model.save(io, "tests/cases/mlp_1_updated_weights");

    const pred_prime = model.forward(.{ 3, 2 });
    const loss_prime = loss_fn(1, pred_prime, .{0.2});
    const loss_grad_prime = loss_grad_fn(1, pred_prime, .{0.2});
    var buf_prime: [30]u8 = undefined;
    const loss_prime_str = try fmt.bufPrint(&buf_prime, "{d}\n", .{loss_prime});
    try file.writeStreamingAll(io, loss_prime_str);

    model.backward(loss_grad_prime);
    optimizer.step();
    try model.save(io, "tests/cases/mlp_1_final_weights");
}

test "mlp 2" {
    const io = testing.io;
    var file: Io.File = try Io.Dir.cwd().createFile(io, "tests/cases/mlp_2_losses", .{});
    defer file.close(io);

    const mlp_config = MLPConfig{
        .in = 1,
        .outs = &.{ 2, 4, 6, 5, 3 },
        .activations = &.{ .ReLU, .ReLU, .ReLU, .ReLU, .None },
        .seed = 1,
    };
    var model: MLP(mlp_config) = .init();
    try model.save(io, "tests/cases/mlp_2_initial_weights");

    const loss_fn = function.MSE;
    const loss_grad_fn = function.MSE_grad;
    var optimizer: Optimizer(.{
        .mlp_config = mlp_config,
        .optimizer = .SGD,
    }) = try .init(&model, 0.0001, 0.9);

    const pred = model.forward(.{2});
    const loss = loss_fn(3, pred, .{ 0.5, 0.2, 0.3 });
    const loss_grad = loss_grad_fn(3, pred, .{ 0.5, 0.2, 0.3 });
    var buf: [30]u8 = undefined;
    const loss_str = try fmt.bufPrint(&buf, "{d}\n", .{loss});
    try file.writeStreamingAll(io, loss_str);

    model.backward(loss_grad);
    optimizer.step();
    optimizer.zero_grad();
    try model.save(io, "tests/cases/mlp_2_updated_weights");

    const pred_prime = model.forward(.{3});
    const loss_prime = loss_fn(3, pred_prime, .{ 0.7, 0.1, 0.2 });
    const loss_grad_prime = loss_grad_fn(3, pred_prime, .{ 0.7, 0.1, 0.2 });
    var buf_prime: [30]u8 = undefined;
    const loss_prime_str = try fmt.bufPrint(&buf_prime, "{d}\n", .{loss_prime});
    try file.writeStreamingAll(io, loss_prime_str);

    model.backward(loss_grad_prime);
    optimizer.step();
    try model.save(io, "tests/cases/mlp_2_final_weights");
}
