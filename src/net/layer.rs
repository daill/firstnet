use ndarray::prelude::*;
use std::fmt;

use crate::net::neuron::Neuron;

pub(crate) trait Layer {
    fn get_weights_size(&self) -> u32;
}

#[derive(Debug, Clone)]
pub struct InputLayer {
    pub inputs: Array1<Neuron>,
}

impl Layer for InputLayer {
    fn get_weights_size(&self) -> u32 {
        self.inputs.len().try_into().unwrap()
    }
}

impl InputLayer {
    fn from_inputs(inputs: Array1<Neuron>) -> Self {
        Self { inputs }
    }

    pub fn new(mut layer_size: u32, bias: bool) -> Self {
        let mut inputs = vec![Neuron::Input::new(); layer_size.try_into().unwrap()];
        if bias {
            inputs.push(Neuron::Bias::new());
        }
        Self { inputs }
    }
}

impl fmt::Display for InputLayer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "neurons: {:?}\n", self.inputs)
    }
}

#[derive(Debug, Clone)]
pub struct OutputLayer {
    pub outputs: Vec<Neuron>,
    pub activation_function: fn(f32) -> f32,
}

impl OutputLayer {
    pub fn new(
        layer_size: u32,
        activation_function: fn(f32) -> f32,
        weight_function: fn(u32) -> Array1<f32>,
        prev_layer: &dyn Layer,
    ) -> Self {
        let weights = weight_function(prev_layer.get_weights_size());
        let mut outputs: Vec<Neuron> = vec![Neuron::new(weights); layer_size.try_into().unwrap()];
        Self {
            outputs: vec![Neuron::new(weights); layer_size.try_into().unwrap()],
            activation_function,
        }
    }
}

impl Layer for OutputLayer {}

impl fmt::Display for OutputLayer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "neurons: {:?}\n", self.outputs)
    }
}

#[derive(Debug, Clone)]
pub struct HiddenLayer {
    pub neurons: Vec<Neuron>,
    pub activation_function: fn(f32) -> f32,
    pub bias: bool,
}

impl HiddenLayer {
    pub fn new(
        layer_size: u32,
        bias: bool,
        activation_function: fn(f32) -> f32,
        weight_function: fn(u32) -> Array1<f32>,
        prev_layer: &dyn Layer,
    ) -> Self {
        let weights = weight_function(prev_layer.get_weights_size());
        let mut neurons: Vec<Neuron> =
            vec![Neuron::new(weights, NeuronKind::Neuron); layer_size.try_into().unwrap()];
        if bias {
            neurons.push(Neuron::new(weights, NeuronKind::Bias));
        }
        Self {
            neurons,
            activation_function,
            bias,
        }
    }
}

impl Layer for HiddenLayer {
    fn get_weights_size(&self) -> u32 {
        self.neurons.len()
    }
}

impl fmt::Display for HiddenLayer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "neurons: {:?}\n", self.neurons)
    }
}

#[cfg(test)]
mod tests {
    use crate::net::{
        activation_functions::{self, sigmoid},
        weight_functions::xavier_init,
    };

    use super::*;

    #[test]
    fn layer_test() {
        let input = InputLayer::new(2, true);
        let a = HiddenLayer::new(5, true, sigmoid, xavier_init, &input);
        println!("{:?}", a);
        assert_eq!(1, 0);
    }
}
