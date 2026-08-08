pub const OptimizerType = enum { SGD, RMSprop, Adam };
pub const OptimizerError = error{ InvalidLRError, InvalidMomentumError };

/// Optimizer allows for the selection of different optimizers, providing unique configurations for each
/// "optim_type" is the optimizer selected
pub fn Optimizer(comptime optimizer_type: OptimizerType) type {
    switch (optimizer_type) {
        .SGD => return struct {
            optimizer_type: OptimizerType = optimizer_type,
            lr: f32,
            momentum: ?f32,

            const Self = @This();
            /// momentum is optional, if not given, optimizer performs stochastic gradient descent,
            /// if given, optimizer performs stochastic gradient descent with momentum
            pub fn init(lr: f32, momentum: ?f32) OptimizerError!Self {
                if (lr < 0) return OptimizerError.InvalidLRError;
                if (momentum) |val| if (val < 0) return OptimizerError.InvalidMomentumError;

                return .{ .lr = lr, .momentum = momentum };
            }
        },
        .RMSprop => {},
        .Adam => {},
    }
}
