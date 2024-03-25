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
    pub input_value: f32,
    pub output_value: f32,
}

#[derive(Clone, Debug)]
pub struct Hidden {
    pub input_value: f32,
    pub output_value: f32,
    pub weights: Array1<f32>,
}

#[derive(Clone, Debug)]
pub struct Bias {
    pub input_value: f32,
    pub output_value: f32,
}

#[derive(Clone, Debug)]
pub struct Output {
    pub input_value: f32,
    pub output_value: f32,
    pub weights: Array1<f32>,
}

impl TryFrom<Neuron> for Input {
    type Error = Neuron;

    fn try_from(other: Neuron) -> Result<Self, Self::Error> {
        match other {
            Neuron::Input(c) => Ok(c),
            a => Err(a),
        }
    }
}

impl TryFrom<Neuron> for Hidden {
    type Error = Neuron;

    fn try_from(other: Neuron) -> Result<Self, Self::Error> {
        match other {
            Neuron::Hidden(c) => Ok(c),
            a => Err(a),
        }
    }
}

impl TryFrom<Neuron> for Output {
    type Error = Neuron;

    fn try_from(other: Neuron) -> Result<Self, Self::Error> {
        match other {
            Neuron::Output(c) => Ok(c),
            a => Err(a),
        }
    }
}

impl Input {
    pub fn new(weights: Array1<f32>) -> Input {
        Input {
            input_value: 0.0,
            output_value: 0.0,
        }
    }
}

impl fmt::Display for Input {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "input_value: {}, output_value: {}",
            self.input_value, self.output_value
        )
    }
}

impl Hidden {
    pub fn new(weights: Array1<f32>) -> Hidden {
        Hidden {
            input_value: 0.0,
            output_value: 0.0,
            weights,
        }
    }

    pub fn init(&mut self, f: fn(u32) -> Array1<f32>) {
        self.weights = f(self.weights.len().try_into().unwrap());
    }
}

impl fmt::Display for Hidden {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "input_value: {}, output_value: {}",
            self.input_value, self.output_value
        )
    }
}

impl Output {
    pub fn new(weights: Array1<f32>) -> Output {
        Output {
            input_value: 0.0,
            output_value: 0.0,
            weights,
        }
    }

    pub fn init(&mut self, f: fn(u32) -> Array1<f32>) {
        self.weights = f(self.weights.len().try_into().unwrap());
    }
}

impl fmt::Display for Output {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "input_value: {}, output_value: {}",
            self.input_value, self.output_value
        )
    }
}

impl Bias {
    pub fn new() -> Bias {
        Bias {
            input_value: 0.0,
            output_value: 1.0,
        }
    }
}

impl fmt::Display for Bias {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "input_value: {}, output_value: {}",
            self.input_value, self.output_value
        )
    }
}
