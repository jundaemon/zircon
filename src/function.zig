const std = @import("std");
const math = std.math;

pub const Activation = enum { None, Tanh, ReLU };

pub fn MSE(comptime out_size: usize, predicted: [out_size]f32, expected: [out_size]f32) f32 {
    if (out_size < 1) @compileError("out_size should be 1 or more");

    var mse: f32 = 0;
    for (0..out_size) |i| mse += math.pow(f32, predicted[i] - expected[i], 2);
    mse /= out_size;

    return mse;
}

pub fn MSE_grad(comptime out_size: usize, predicted: [out_size]f32, expected: [out_size]f32) [out_size]f32 {
    if (out_size < 1) @compileError("out_size should be 1 or more");

    var mse_grad: [out_size]f32 = undefined;
    for (0..out_size) |i| mse_grad[i] = (predicted[i] - expected[i]) * 2 / out_size;

    return mse_grad;
}
