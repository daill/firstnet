/*
 * Regression - Linear Activation Function
 * Binary Classification—Sigmoid/Logistic Activation Function
 * Multiclass Classification—Softmax
 * Multilabel Classification—Sigmoid
 */

// should not be used in hidden layers
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn sigmoid_derivative(x: f32) -> f32 {
    x * (1.0 - x)
}

pub fn tanh(x: f32) -> f32 {
    (1.0 - (-x).exp()) / (1.0 + (-x).exp())
}

pub fn tanh_derivative(x: f32) -> f32 {
    1.0 - x * x
}

// should only be used in hidden layer
pub fn relu(x: f32) -> f32 {
    if x < 0.0 {
        0.0
    } else {
        x
    }
}

pub fn relu_derivative(x: f32) -> f32 {
    if x < 0.0 {
        0.0
    } else {
        1.0
    }
}

// should be used in when the depth of the network is > 40
pub fn swish(x: f32) -> f32 {
    x * sigmoid(x)
}

pub fn nop(x: f32) -> f32 {
    x
}

pub fn nop_derivative(x: f32) -> f32 {
    1.0
}