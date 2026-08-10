const std = @import("std");
const Io = std.Io;
const mem = std.mem;
const Random = std.Random;

const layer = @import("layer");
const Layer = layer.Layer;

const conf = @import("conf");
const MLPConfig = conf.MLPConfig;

pub fn MLP(comptime config: MLPConfig) type {
    config.check();
    const Layers = config.create_layer_types();
    const Ins = config.create_in_types();
    const Outs = config.create_out_types();
    const n = config.outs.len;
    const dimensions = [1]usize{config.in} ++ config.outs;

    return struct {
        layers: Layers,

        const Self = @This();
        pub fn init() Self {
            var prng: Random.DefaultPrng = .init(config.seed);
            const rand = prng.random();

            var layers: Layers = undefined;
            inline for (0..n) |i| layers[i] = .init(rand);

            return .{ .layers = layers };
        }

        pub fn load(io: Io, path: []const u8) !Self {
            var file: Io.File = try Io.Dir.cwd().openFile(io, path, .{});
            defer file.close(io);

            var buf: [4_096]u8 = undefined;
            var reader: Io.File.Reader = file.reader(io, &buf);
            const interface = &reader.interface;

            var layers: Layers = undefined;
            inline for (0..n) |i| {
                const in = dimensions[i];
                const out = dimensions[i + 1];

                var weight_bytes: [out * in * 4]u8 = undefined;
                try interface.readSliceAll(&weight_bytes);

                var bias_bytes: [out * 4]u8 = undefined;
                try interface.readSliceAll(&bias_bytes);

                layers[i] = .load(
                    mem.bytesToValue([out][in]f32, &weight_bytes),
                    mem.bytesToValue([out]f32, &bias_bytes),
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

            inline for (0..n) |i| {
                const out = dimensions[i + 1];

                var weight: [out][dimensions[i]]f32 = undefined;
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

        pub fn forward(self: *Self, input: [config.in]f32) [dimensions[n]]f32 {
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

        pub fn backward(self: *Self, loss_grad: [dimensions[n]]f32) void {
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

        pub fn zero_grad(self: *Self) void {
            inline for (0..n) |i| self.layers[i].zero_grad();
        }
    };
}

test {
    const config = MLPConfig{ .in = 1, .outs = &.{ 2, 3, 4 }, .activations = &.{ .ReLU, .Tanh, .ReLU }, .seed = 1 };
    const model = MLP(config).init();
    std.debug.print("{any}", .{model.layers});
}
