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
    f: []const Activation,
    seed: u64,

    pub fn check(comptime self: MLPConfig) void {
        const in = self.in;
        const outs = self.outs;
        const f = self.f;

        if (in == 0) @compileError("in should be 1 or more");
        if (outs.len == 0 or f.len == 0) @compileError("number of layers should be 1 or more");
        if (outs.len != f.len) @compileError("outs and f should have the same length");
        for (outs) |out| if (out == 0) @compileError("number of neurons in each layer should be 1 or more");
    }
};

pub fn MLP(comptime mlp_config: MLPConfig) type {
    mlp_config.check();
    const num_layers = mlp_config.outs.len;
    const dimensions = [1]usize{mlp_config.in} ++ mlp_config.outs;

    var layer_types: [num_layers]type = undefined;
    var Y_types: [num_layers]type = undefined;
    var dL_dX_types: [num_layers]type = undefined;
    for (0..num_layers) |i| {
        const in = dimensions[i];
        const out = dimensions[i + 1];
        const f = mlp_config.f[i];

        layer_types[i] = Layer(in, out, f);
        Y_types[i] = [out]f32;
        dL_dX_types[i] = [in]f32;
    }

    const Layers = @Tuple(&layer_types);
    const Ys = @Tuple(&Y_types);
    const dL_dXs = @Tuple(&dL_dX_types);

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

                var layer_W_bytes: [out * in * 4]u8 = undefined;
                try interface.readSliceAll(&layer_W_bytes);

                var layer_B_bytes: [out * 4]u8 = undefined;
                try interface.readSliceAll(&layer_B_bytes);

                layers[i] = .load(
                    mem.bytesToValue([out][in]f32, &layer_W_bytes),
                    mem.bytesToValue([out]f32, &layer_B_bytes),
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

                var layer_W: [out][in]f32 = undefined;
                var layer_B: [out]f32 = undefined;
                for (0..out) |j| {
                    const neuron_W = self.layers[i].W[j];
                    const neuron_b = self.layers[i].B[j];

                    layer_W[j] = neuron_W;
                    layer_B[j] = neuron_b;
                }

                try interface.writeAll(mem.asBytes(&layer_W));
                try interface.writeAll(mem.asBytes(&layer_B));
            }

            try interface.flush();
        }

        pub fn forward(self: *Self, X: [mlp_config.in]f32) [dimensions[num_layers]]f32 {
            var incremental_Y: Ys = undefined;
            inline for (0..num_layers) |i| {
                if (i == 0) {
                    incremental_Y[i] = self.layers[i].forward(X);
                } else {
                    incremental_Y[i] = self.layers[i].forward(incremental_Y[i - 1]);
                }
            }

            return incremental_Y[num_layers - 1];
        }

        pub fn backward(self: *Self, dL_dy_cap: [dimensions[num_layers]]f32) void {
            var incremental_dL_dX: dL_dXs = undefined;
            inline for (0..num_layers) |i| {
                const reverse_i = num_layers - i - 1;
                if (reverse_i == num_layers - 1) {
                    incremental_dL_dX[reverse_i] = self.layers[reverse_i].backward(dL_dy_cap);
                } else {
                    incremental_dL_dX[reverse_i] = self.layers[reverse_i].backward(incremental_dL_dX[reverse_i + 1]);
                }
            }
        }
    };
}

test "mlp training loop 1" {
    const io = testing.io;
    var file: Io.File = try Io.Dir.cwd().createFile(io, "tests/cases/mlp_1_losses", .{});
    defer file.close(io);

    const mlp_config = MLPConfig{
        .in = 2,
        .outs = &.{ 3, 5, 7, 4, 1 },
        .f = &.{ .Tanh, .Tanh, .Tanh, .Tanh, .None },
        .seed = 1,
    };
    var model: MLP(mlp_config) = .init();
    try model.save(io, "tests/cases/mlp_1_initial_weights");

    const loss_fn = function.MSE;
    const loss_grad_fn = function.MSE_grad;
    var optimizer: Optimizer(.{
        .mlp_config = mlp_config,
        .optimizer = .SGD,
    }) = try .init(&model, .{ .lr = 1e-4 });

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

test "mlp training loop 2" {
    const io = testing.io;
    var file: Io.File = try Io.Dir.cwd().createFile(io, "tests/cases/mlp_2_losses", .{});
    defer file.close(io);

    const mlp_config = MLPConfig{
        .in = 1,
        .outs = &.{ 2, 4, 6, 5, 3 },
        .f = &.{ .ReLU, .ReLU, .ReLU, .ReLU, .None },
        .seed = 1,
    };
    var model: MLP(mlp_config) = .init();
    try model.save(io, "tests/cases/mlp_2_initial_weights");

    const loss_fn = function.MSE;
    const loss_grad_fn = function.MSE_grad;
    var optimizer: Optimizer(.{
        .mlp_config = mlp_config,
        .optimizer = .SGD,
    }) = try .init(&model, .{ .lr = 1e-4, .momentum = 0.9 });

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

test "mlp training loop 3" {
    const io = testing.io;
    var file: Io.File = try Io.Dir.cwd().createFile(io, "tests/cases/mlp_3_losses", .{});
    defer file.close(io);

    const mlp_config = MLPConfig{
        .in = 3,
        .outs = &.{ 9, 7, 5, 3, 1 },
        .f = &.{ .ReLU, .Tanh, .ReLU, .Tanh, .None },
        .seed = 1,
    };
    var model: MLP(mlp_config) = .init();
    try model.save(io, "tests/cases/mlp_3_initial_weights");

    const loss_fn = function.MSE;
    const loss_grad_fn = function.MSE_grad;
    var optimizer: Optimizer(.{
        .mlp_config = mlp_config,
        .optimizer = .RMSprop,
    }) = try .init(&model, .{ .lr = 1e-4, .decay_rate = 0.9 });

    const pred = model.forward(.{ 1, 2, 3 });
    const loss = loss_fn(1, pred, .{0.4});
    const loss_grad = loss_grad_fn(1, pred, .{0.4});
    var buf: [30]u8 = undefined;
    const loss_str = try fmt.bufPrint(&buf, "{d}\n", .{loss});
    try file.writeStreamingAll(io, loss_str);

    model.backward(loss_grad);
    optimizer.step();
    optimizer.zero_grad();
    try model.save(io, "tests/cases/mlp_3_updated_weights");

    const pred_prime = model.forward(.{ 5, 4, 3 });
    const loss_prime = loss_fn(1, pred_prime, .{0.3});
    const loss_grad_prime = loss_grad_fn(1, pred_prime, .{0.3});
    var buf_prime: [30]u8 = undefined;
    const loss_prime_str = try fmt.bufPrint(&buf_prime, "{d}\n", .{loss_prime});
    try file.writeStreamingAll(io, loss_prime_str);

    model.backward(loss_grad_prime);
    optimizer.step();
    try model.save(io, "tests/cases/mlp_3_final_weights");
}
