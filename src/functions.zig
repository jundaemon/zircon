const std = @import("std");
const math = std.math;

pub const Activation = enum { None, Tanh, ReLU };

pub fn MSE(comptime n: usize, predicted: [n]f32, expected: [n]f32) f32 {
    if (n < 1) @compileError("n should be 1 or more");

    var mse: f32 = 0;
    for (0..n) |i| mse += math.pow(f32, predicted[i] - expected[i], 2);
    mse /= n;

    return mse;
}

pub fn MSE_grad(comptime n: usize, predicted: [n]f32, expected: [n]f32) [n]f32 {
    if (n < 1) @compileError("n should be 1 or more");

    var mse_grad: [n]f32 = 0;
    for (0..n) |i| mse_grad[i] = (predicted[i] - expected[i]) * 2 / n;

    return mse_grad;
}
