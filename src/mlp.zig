const std = @import("std");
const Io = std.Io;
const mem = std.mem;
const Random = std.Random;

const function = @import("function.zig");
const Activation = function.Activation;

const layer = @import("layer.zig");
const Layer = layer.Layer;

/// configurations for MLP
///
/// properties:
/// in -> the size of input into the MLP
/// outs -> the number of neurons in each layer
/// f -> the activation functions for neurons in each layer
/// seed -> for reproducible weight initialization
pub const MLPConfig = struct {
    in: usize,
    outs: []const usize,
    f: []const Activation,
    seed: u64,

    /// a comptime check for struct properties, mainly used in development of the library
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

/// consolidation of all layers in the MLP
///
/// arguments:
/// mlp_config -> comptime data for construction of MLP architecture
pub fn MLP(comptime mlp_config: MLPConfig) type {
    mlp_config.check();
    const num_layers = mlp_config.outs.len;
    const dimensions = [1]usize{mlp_config.in} ++ mlp_config.outs;

    var layer_types: [num_layers]type = undefined;
    var Y_types: [num_layers]type = undefined;
    for (0..num_layers) |i| {
        const in = dimensions[i];
        const out = dimensions[i + 1];
        const f = mlp_config.f[i];

        layer_types[i] = Layer(in, out, f);
        Y_types[i] = [out]f32;
    }

    const Layers = @Tuple(&layer_types);
    const Ys = @Tuple(&Y_types);

    return struct {
        layers: Layers,

        const Self = @This();
        /// initializes all weights and biases using each layer's .init method and returns an MLP struct instance
        pub fn init() Self {
            var prng: Random.DefaultPrng = .init(mlp_config.seed);
            const rand = prng.random();

            var layers: Layers = undefined;
            inline for (0..num_layers) |i| layers[i] = .init(rand);

            return .{ .layers = layers };
        }

        /// loads saved weights and biases of a model with the same architecture and returns a MLP struct instance
        ///
        /// attempting to load a model with a different architecture or any other file that was not generated using
        /// this library will lead to errors or garbage weights
        ///
        /// arguments:
        /// io -> Zig's io
        /// path -> relative path to the file containing model weights and biases
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

        /// saves all weights and biases of a model contiguously as bytes in a file
        ///
        /// arguments:
        /// io -> Zig's io
        /// path -> relative path of file to save weights and biases in
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

        /// does a forward pass through all layers in the model and returns the output from the final layer
        ///
        /// arguments:
        /// X -> input to the neurons in the first layer of the model
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
    };
}
