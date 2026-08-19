const std = @import("std");
const math = std.math;
const Random = std.Random;

const function = @import("function");
const Activation = function.Activation;

fn custom_rand_f32(rand: Random, bound: f32) f32 {
    return rand.float(f32) * bound * 2 - bound;
}

/// a consolidation of all data related to a layer and its neurons including:
/// weights, gradient of weights, biases, gradients of biases, the output and input at a certain epoch
///
/// arguments:
/// in -> number of in channels to the layer
/// out -> number of out channels to the layer
/// f -> activation function used for all neurons in the layer
pub fn Layer(comptime in: usize, comptime out: usize, comptime f: Activation) type {
    return struct {
        W: [out][in]f32,
        dL_dW: [out][in]f32 = [_][in]f32{@splat(0)} ** out,
        B: [out]f32,
        dL_dB: [out]f32 = @splat(0),
        Y: [out]f32 = @splat(0),
        X: [in]f32 = @splat(0),

        const Self = @This();
        /// initializes all weights and biases of neurons within the layer and returns a Layer struct instance
        ///
        /// if the layer uses a symmetric activation function, then weights are initialized using Xavier uniform distribution
        /// if the layer uses an asymmetric activation function, then weights are initialized using He uniform distribution in fan in mode
        /// biases are initialized using LeCun uniform distribution regardless
        ///
        /// arguments:
        /// rand -> Zig's Random
        pub fn init(rand: Random) Self {
            const w_bound = switch (f) {
                .None, .Tanh => math.sqrt(6 / @as(f32, in + out)),
                .ReLU => math.sqrt(6 / @as(f32, in)),
            };
            const b_bound = 1 / math.sqrt(@as(f32, in));

            var layer_W: [out][in]f32 = undefined;
            var layer_B: [out]f32 = undefined;
            for (0..out) |i| {
                var neuron_W: [in]f32 = undefined;
                for (0..in) |j| neuron_W[j] = custom_rand_f32(rand, w_bound);

                layer_W[i] = neuron_W;
                layer_B[i] = custom_rand_f32(rand, b_bound);
            }

            return .{ .W = layer_W, .B = layer_B };
        }

        pub fn load(W: [out][in]f32, B: [out]f32) Self {
            return .{ .W = W, .B = B };
        }

        /// performs a forward pass through the layer and returns the output
        /// this method also saves the input to the layer for use in loss.backward()
        ///
        /// arguments:
        /// X -> input to the neurons in the layer
        pub fn forward(self: *Self, X: [in]f32) [out]f32 {
            self.X = X;
            self.Y = @splat(0);

            // z = wx + b, y = f(z)
            for (0..out) |i| {
                for (0..in) |j| self.Y[i] += X[j] * self.W[i][j];
                self.Y[i] += self.B[i];
            }
            switch (f) {
                .None => {},
                .Tanh => {
                    for (0..out) |i| {
                        const z = self.Y[i];
                        self.Y[i] = math.tanh(z);
                    }
                },
                .ReLU => {
                    for (0..out) |i| {
                        const z = self.Y[i];
                        self.Y[i] = if (z > 0) z else 0;
                    }
                },
            }

            return self.Y;
        }

        /// calculates derivatives of loss wrt the weights, biases and inputs to the layer using chain rule
        ///
        /// arguments:
        /// i -> the 0-indexed neuron number
        /// dL_dz -> the derivative of loss wrt the pre-activation output of the neuron
        /// dL_dX -> the derivative of loss wrt the input to the neuron, to be accumulated through all neurons
        fn backward_helper(self: *Self, i: usize, dL_dz: f32, dL_dX: []f32) void {
            for (0..in) |j| {
                self.dL_dW[i][j] += self.X[j] * dL_dz;
                dL_dX[j] += self.W[i][j] * dL_dz;
            }
            self.dL_dB[i] += dL_dz;
        }

        /// performs backpropagation through the layer and returns the derivative of loss wrt the input
        ///
        /// arguments:
        /// dL_dY -> derivative of loss wrt the output of neurons in the layer
        pub fn backward(self: *Self, dL_dY: [out]f32) [in]f32 {
            var dL_dX: [in]f32 = @splat(0);
            switch (f) {
                .None => {
                    for (0..out) |i| self.backward_helper(i, dL_dY[i], &dL_dX);
                },
                .Tanh => {
                    for (0..out) |i| {
                        // derivative of y wrt to z in this case is the derivative of tanh = 1 - tanh^2(z)
                        const dL_dz = dL_dY[i] * (1 - math.pow(f32, self.Y[i], 2));
                        self.backward_helper(i, dL_dz, &dL_dX);
                    }
                },
                .ReLU => {
                    for (0..out) |i| {
                        // derivative of y wrt to z in this case is the derivative of relu = I(z > 0)
                        const dL_dz = dL_dY[i] * if (self.Y[i] > 0) @as(f32, 1) else 0;
                        self.backward_helper(i, dL_dz, &dL_dX);
                    }
                },
            }

            return dL_dX;
        }
    };
}
