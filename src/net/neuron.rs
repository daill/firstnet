use ndarray::prelude::*;
use std::fmt;

#[derive(Clone, Debug)]
pub enum Neuron {
    Input(Input),
    Bias(Bias),
    Hidden(Hidden),
    Output(Output),
}

#[derive(Clone, Debug)]
pub struct Input {
    pub value: f32,
}

#[derive(Clone, Debug)]
pub struct Hidden {
    pub value: f32,
    pub weights: Array1<f32>,
}

#[derive(Clone, Debug)]
pub struct Bias {
    pub value: f32,
}

#[derive(Clone, Debug)]
pub struct Output {
    pub value: f32,
    pub weights: Array1<f32>,
}

impl Input {
    pub fn new(weights: Array1<f32>) -> Input {
        Input { value: 0.0 }
    }
}

impl fmt::Display for Input {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "value: {}", self.value)
    }
}

impl Hidden {
    pub fn new(weights: Array1<f32>) -> Hidden {
        Hidden {
            value: 0.0,
            weights,
        }
    }

    pub fn init(&mut self, f: fn(u32) -> Array1<f32>) {
        self.weights = f(self.weights.len().try_into().unwrap());
    }
}

impl fmt::Display for Hidden {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "value: {}", self.value)
    }
}

impl Output {
    pub fn new(weights: Array1<f32>) -> Output {
        Output {
            value: 0.0,
            weights,
        }
    }

    pub fn init(&mut self, f: fn(u32) -> Array1<f32>) {
        self.weights = f(self.weights.len().try_into().unwrap());
    }
}

impl fmt::Display for Output {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "value: {}", self.value)
    }
}

impl Bias {
    pub fn new() -> Bias {
        Bias { value: 1.0 }
    }
}

impl fmt::Display for Bias {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "value: {}", self.value)
    }
}
