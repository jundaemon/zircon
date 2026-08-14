const std = @import("std");
const math = std.math;

pub const Activation = enum { None, Tanh, ReLU };

pub fn MSE(comptime out: usize, y: [out]f32, y_cap: [out]f32) f32 {
    if (out < 1) @compileError("out should be 1 or more");

    var mse: f32 = 0;
    for (0..out) |i| mse += math.pow(f32, y[i] - y_cap[i], 2);
    mse /= out;

    return mse;
}

pub fn MSE_grad(comptime out: usize, y: [out]f32, y_cap: [out]f32) [out]f32 {
    if (out < 1) @compileError("out should be 1 or more");

    var mse_grad: [out]f32 = undefined;
    for (0..out) |i| mse_grad[i] = (y[i] - y_cap[i]) * 2 / out;

    return mse_grad;
}
