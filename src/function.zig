const std = @import("std");
const math = std.math;

const mlp = @import("mlp");
const MLPConfig = mlp.MLPConfig;
const MLP = mlp.MLP;

pub const Activation = enum { None, Tanh, ReLU };
pub const LossFunction = enum { MSE };

pub fn Loss(comptime mlp_config: MLPConfig, comptime loss_function: LossFunction) type {
    mlp_config.check();
    const num_layers = mlp_config.outs.len;
    const dimensions = [1]usize{mlp_config.in} ++ mlp_config.outs;
    const model_out = dimensions[num_layers];

    var dL_dX_types: [num_layers]type = undefined;
    for (0..num_layers) |i| dL_dX_types[i] = [dimensions[i]]f32;

    const dL_dXs = @Tuple(&dL_dX_types);

    return struct {
        model_ptr: *MLP(mlp_config),

        const Self = @This();
        pub fn init(model_ptr: *MLP(mlp_config)) Self {
            return .{ .model_ptr = model_ptr };
        }

        pub fn eval(self: Self, y: [model_out]f32, y_cap: [model_out]f32) struct {
            model_ptr: *MLP(mlp_config),
            item: f32,
            dL_dy_cap: [model_out]f32,

            const Self_ = @This();
            pub fn backward(self_: Self_) void {
                const model_ptr = self_.model_ptr;
                const dL_dy_cap = self_.dL_dy_cap;

                var incremental_dL_dX: dL_dXs = undefined;
                inline for (0..num_layers) |i| {
                    const reverse_i = num_layers - i - 1;
                    if (reverse_i == num_layers - 1) {
                        incremental_dL_dX[reverse_i] = model_ptr.layers[reverse_i].backward(dL_dy_cap);
                    } else {
                        incremental_dL_dX[reverse_i] = model_ptr.layers[reverse_i].backward(incremental_dL_dX[reverse_i + 1]);
                    }
                }
            }
        } {
            switch (loss_function) {
                .MSE => {
                    var mse: f32 = 0;
                    var dL_dy_cap: [model_out]f32 = undefined;
                    for (0..model_out) |i| {
                        const ae = y[i] - y_cap[i];

                        mse += math.pow(f32, ae, 2);
                        dL_dy_cap[i] = ae * 2 / model_out;
                    }
                    mse /= model_out;

                    return .{
                        .model_ptr = self.model_ptr,
                        .item = mse,
                        .dL_dy_cap = dL_dy_cap,
                    };
                },
            }
        }
    };
}
