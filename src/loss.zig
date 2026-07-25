const std = @import("std");

const math = std.math;

fn MSE(predicted: f32, actual: f32) f32 {
    return math.pow(f32, predicted - actual, 2);
}

fn MSEGrad(predicted: f32, actual: f32) f32 {
    return 2 * (predicted - actual);
}
