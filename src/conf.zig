const functions = @import("functions");
const Activation = functions.Activation;

const layer = @import("layer");
const Layer = layer.Layer;

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

    pub fn create_layer_types(comptime self: MLPConfig) type {
        const dimensions = [1]usize{self.in} ++ self.outs;
        var layer_types: [self.outs.len]type = undefined;
        for (0..self.outs.len) |i| layer_types[i] = Layer(dimensions[i], dimensions[i + 1], self.activations[i]);

        return @Tuple(&layer_types);
    }

    pub fn create_in_types(comptime self: MLPConfig) type {
        const dimensions = [1]usize{self.in} ++ self.outs;
        var in_types: [self.outs.len]type = undefined;
        for (0..self.outs.len) |i| in_types[i] = [dimensions[i]]f32;

        return @Tuple(&in_types);
    }

    pub fn create_out_types(comptime self: MLPConfig) type {
        var out_types: [self.outs.len]type = undefined;
        for (0..self.outs.len) |i| out_types[i] = [self.outs[i]]f32;

        return @Tuple(&out_types);
    }
};
