const std = @import("std");
const math = std.math;

/// returns mean squared error loss, followed by the derivative of mean squared error loss
pub fn MSE(comptime n: usize, predicted: [n]f32, expected: [n]f32) struct { f32, [n]f32 } {
    if (n < 1) @compileError("output should be of size 1 or larger");

    var mse: f32 = 0;
    var mse_grad: [n]f32 = undefined;
    for (0..n) |i| {
        const ae = predicted[i] - expected[i];
        mse += math.pow(f32, ae, 2);
        mse_grad[i] = ae * 2 / n;
    }
    mse /= n;

    return .{ mse, mse_grad };
}
