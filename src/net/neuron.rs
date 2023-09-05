use ndarray::prelude::*;
use std::fmt;

pub trait Element {}

#[derive(Clone, Debug)]
pub struct Neuron {
    pub value: f32,
    pub weights: Array1<f32>,
}

impl Element for Neuron {}

impl Neuron {
    pub fn new(weights: Array1<f32>) -> Neuron {
        Neuron {
            value: 0.0,
            weights,
        }
    }

    pub fn from_value(value: f32, weights_size: u32) -> Neuron {
        Neuron {
            value,
            weights: Array1::zeros(weights_size as usize),
        }
    }

    pub fn init(&mut self, f: fn(u32) -> Array1<f32>) {
        self.weights = f(self.weights.len().try_into().unwrap());
    }
}

impl fmt::Display for Neuron {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "value: {}", self.value)
    }
}

#[derive(Clone, Debug)]
pub struct Bias {
    pub value: f32,
}

impl Element for Bias {}

impl Bias {
    pub fn new() -> Bias {
        Bias { value: 1.0 }
    }
}

#[derive(Clone, Debug)]
pub struct Input {
    pub value: f32,
}

impl Element for Input {}

impl Input {
    pub fn new() -> Input {
        Input { value: 0.0 }
    }
}
