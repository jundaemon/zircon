const conf = @import("conf");
const MLPConfig = conf.MLPConfig;

pub const OptimizerType = enum { SGD, RMSprop, Adam };

pub fn Optimizer1(comptime optimizer_type: OptimizerType) type {
    switch (optimizer_type) {
        .SGD => return struct {
            lr: f32,
            momentum: ?f32,

            const Self = @This();
            /// momentum is optional, if not given, optimizer performs stochastic gradient descent,
            /// if given, optimizer performs stochastic gradient descent with momentum
            pub fn init(comptime lr: f32, comptime momentum: ?f32) Self {
                if (lr < 0) @compileError("learning rate should be 0 or larger");
                if (momentum) |val| if (val < 0) @compileError("momentum should be 0 or larger");

                return .{ .lr = lr, .momentum = momentum };
            }
        },
        .RMSprop => return struct { lr: f32 },
        .Adam => return struct { lr: f32 },
    }
}
