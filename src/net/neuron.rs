use ndarray::prelude::*;
use std::fmt;
use rand::distributions::WeightedError;

#[derive(Clone, Debug)]
pub enum Neuron {
    Input(Input),
    Bias(Bias),
    Hidden(Hidden),
    Output(Output),
}

pub trait NeuronBase {
    fn get_input_value(&self) -> f32;
    fn set_input_value(&mut self, input_value: f32);

    fn get_output_value(&self) -> f32;

    fn set_output_value(&mut self, output_value: f32);

    fn get_mut_weights(&mut self) -> &mut Array1<f32>;
    fn get_weights(&self) -> &Array1<f32>;
}

impl NeuronBase for Neuron {
    fn get_input_value(&self) -> f32 {
        match self {
            Neuron::Input(i) => i.input_value,
            Neuron::Bias(b) => b.input_value,
            Neuron::Hidden(h) => h.input_value,
            Neuron::Output(o) => o.input_value,
        }
    }
    fn set_input_value(&mut self, input_value: f32) {
        match self {
            Neuron::Input(i) => i.input_value = input_value,
            Neuron::Bias(b) => b.input_value = input_value,
            Neuron::Hidden(h) => h.input_value = input_value,
            Neuron::Output(o) => o.input_value = input_value,
        }
    }
    fn get_output_value(&self) -> f32 {
        match self {
            Neuron::Input(i) => i.output_value,
            Neuron::Bias(b) => b.output_value,
            Neuron::Hidden(h) => h.output_value,
            Neuron::Output(o) => o.output_value,
        }
    }
    fn set_output_value(&mut self, output_value: f32) {
        match self {
            Neuron::Input(i) => i.output_value = output_value,
            Neuron::Bias(b) => b.output_value = output_value,
            Neuron::Hidden(h) => h.output_value = output_value,
            Neuron::Output(o) => o.output_value = output_value,
        }
    }
    fn get_mut_weights(&mut self) -> &mut Array1<f32> {
        match self {
            Neuron::Input(i) => i.get_mut_weights(),
            Neuron::Bias(b) => b.get_mut_weights(),
            Neuron::Hidden(h) => h.get_mut_weights(),
            Neuron::Output(o) => o.get_mut_weights(),
        }
    }
    fn get_weights(&self) -> &Array1<f32> {
        match self {
            Neuron::Input(i) => i.get_weights(),
            Neuron::Bias(b) => b.get_weights(),
            Neuron::Hidden(h) => h.get_weights(),
            Neuron::Output(o) => o.get_weights(),
        }
    }
}


#[derive(Clone, Debug)]
pub struct Input {
    pub input_value: f32,
    pub output_value: f32,
    pub weights: Array1<f32>,
}

#[derive(Clone, Debug)]
pub struct Hidden {
    pub input_value: f32,
    pub output_value: f32,
    pub weights: Array1<f32>,
}

impl NeuronBase for Hidden {
    fn get_input_value(&self) -> f32 {
        self.input_value
    }
    fn set_input_value(&mut self, input_value: f32) {
        self.input_value = input_value;
    }
    fn get_output_value(&self) -> f32 {
        self.output_value
    }
    fn set_output_value(&mut self, output_value: f32) {
        self.output_value = output_value;
    }
    fn get_mut_weights(&mut self) -> &mut Array1<f32> {
        &mut self.weights
    }

    fn get_weights(&self) -> &Array1<f32> {
        &self.weights
    }
}


impl NeuronBase for Input {
    fn get_input_value(&self) -> f32 {
        self.input_value
    }
    fn set_input_value(&mut self, input_value: f32) {
        self.input_value = input_value;
    }
    fn get_output_value(&self) -> f32 {
        self.output_value
    }
    fn set_output_value(&mut self, output_value: f32) {
        self.output_value = output_value;
    }
    fn get_mut_weights(&mut self) -> &mut Array1<f32> {
        &mut self.weights
    }

    fn get_weights(&self) -> &Array1<f32> {
        &self.weights
    }
}

#[derive(Clone, Debug)]
pub struct Bias {
    pub input_value: f32,
    pub output_value: f32,
    pub weights: Array1<f32>,
}


#[derive(Clone, Debug)]
pub struct Output {
    pub input_value: f32,
    pub output_value: f32,
    pub weights: Array1<f32>,
}

impl NeuronBase for Output {
    fn get_input_value(&self) -> f32 {
        self.input_value
    }
    fn set_input_value(&mut self, input_value: f32) {
        self.input_value = input_value;
    }
    fn get_output_value(&self) -> f32 {
        self.output_value
    }
    fn set_output_value(&mut self, output_value: f32) {
        self.output_value = output_value;
    }
    fn get_mut_weights(&mut self) -> &mut Array1<f32> {
        &mut self.weights
    }

    fn get_weights(&self) -> &Array1<f32> {
        &self.weights
    }
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
            weights: array![0.0],
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
            weights: array![0.0],
        }
    }
}

impl NeuronBase for Bias {
    fn get_input_value(&self) -> f32 {
        self.input_value
    }

    fn set_input_value(&mut self, input_value: f32) {
        self.input_value = input_value;
    }

    fn get_output_value(&self) -> f32 {
        self.output_value
    }

    fn set_output_value(&mut self, output_value: f32) {
        self.output_value = output_value;
    }

    fn get_mut_weights(&mut self) -> &mut Array1<f32> {
        &mut self.weights
    }

    fn get_weights(&self) -> &Array1<f32> {
        &self.weights
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
