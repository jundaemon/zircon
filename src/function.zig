const std = @import("std");
const math = std.math;

const mlp = @import("mlp");
const MLPConfig = mlp.MLPConfig;
const MLP = mlp.MLP;

pub const Activation = enum { None, Tanh, ReLU };
pub const LossFunction = enum { MSE };
/// configurations for Loss
///
/// properties:
/// mlp_config -> reference MLP architecture to build struct around
/// loss_function -> user selected loss function
pub const LossConfig = struct {
    mlp_config: MLPConfig,
    loss_function: LossFunction,
};

/// uses model architecture to build an interface for loss calculation and model backpropagation
///
/// arguments:
/// loss_config -> comptime configurations containing data like parent MLP architecture and user selected loss function
pub fn Loss(comptime loss_config: LossConfig) type {
    const mlp_config = loss_config.mlp_config;

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

        /// evaluates both the loss and derivative of loss wrt predicted value and returns a anonymous struct instance
        /// this anonymous struct instance allows users to retrieve loss through the property .item and perform backpropagation through .backward
        ///
        /// arguments:
        /// y -> the expected output
        /// y_cap -> the prediction made by the model
        pub fn eval(self: Self, y: [model_out]f32, y_cap: [model_out]f32) struct {
            model_ptr: *MLP(mlp_config),
            item: f32,
            dL_dy_cap: [model_out]f32,

            const Self_ = @This();
            pub fn backward(self_: Self_) void {
                const model_ptr = self_.model_ptr;

                var incremental_dL_dX: dL_dXs = undefined;
                inline for (0..num_layers) |i| {
                    const reverse_i = num_layers - i - 1;
                    if (reverse_i == num_layers - 1) {
                        incremental_dL_dX[reverse_i] = model_ptr.layers[reverse_i].backward(self_.dL_dy_cap);
                    } else {
                        incremental_dL_dX[reverse_i] = model_ptr.layers[reverse_i].backward(incremental_dL_dX[reverse_i + 1]);
                    }
                }
            }
        } {
            switch (loss_config.loss_function) {
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
