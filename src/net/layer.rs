use std::any::Any;
use ndarray::prelude::*;
use std::fmt;

use crate::net::neuron::Neuron;

use super::neuron::{Bias, Hidden, Input, Output};

pub trait Layer {
    fn weights_size(&self) -> u32;
    fn values_as_arr(&self) -> Array1<f32>;

    fn len(&self) -> usize;
    fn get(&self, index: usize) -> Option<&Neuron>;

    fn get_mut(&mut self, index: usize) -> Option<&mut Neuron>;

    fn get_all(&self) -> &Array1<Neuron>;

    fn get_all_mut(&mut self) -> &mut Array1<Neuron>;

    fn as_any(&self) -> &dyn Any;

    fn get_activation_derivation(&self) -> fn(f32) -> f32;
}

#[derive(Debug, Clone)]
pub struct InputLayer {
    pub inputs: Array1<Neuron>,
}

impl Layer for InputLayer {
    fn weights_size(&self) -> u32 {
        self.inputs.len().try_into().unwrap()
    }

    fn values_as_arr(&self) -> Array1<f32> {
        Array1::from_vec(
            self.inputs
                .iter()
                .map(|n| match n {
                    Neuron::Input(i) => i.output_value,
                    _ => 0.0,
                })
                .collect(),
        )
    }

    fn len(&self) -> usize {
        self.inputs.len()
    }

    fn get(&self, index: usize) -> Option<&Neuron> {
        self.inputs.get(index)
    }

    fn get_mut(&mut self, index: usize) -> Option<&mut Neuron> {
        self.inputs.get_mut(index)
    }

    fn get_all(&self) -> &Array1<Neuron> {
        &self.inputs
    }

    fn get_all_mut(&mut self) -> &mut Array1<Neuron> {
        &mut self.inputs
    }
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_activation_derivation(&self) -> fn(f32) -> f32 {
        |x| x
    }
}

impl InputLayer {
    fn from_inputs(inputs: Array1<Neuron>) -> Self {
        Self { inputs }
    }

    pub fn new(mut layer_size: u32, bias: bool) -> Self {
        let mut inputs = vec![Neuron::Input(Input { input_value: 0.0, output_value: 0.0 }); layer_size.try_into().unwrap()];
        if bias {
            inputs.push(Neuron::Bias(Bias { input_value: 1.0, output_value: 1.0 }));
        }
        Self {
            inputs: Array1::from_vec(inputs),
        }
    }

    pub fn set_inputs(&mut self, input_values: Vec<f32>) {
        self.inputs = Array1::from_vec(
            input_values
                .into_iter()
                .map(|n| Neuron::Input(Input { input_value: n, output_value: n }))
                .collect(),
        );
    }
}

impl fmt::Display for InputLayer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "neurons: {:?}\n", self.inputs)
    }
}

#[derive(Debug, Clone)]
pub struct OutputLayer {
    pub outputs: Array1<Neuron>,
    pub activation_function: fn(f32) -> f32,
    pub activation_derivation: fn(f32) -> f32,
}

impl OutputLayer {
    pub fn new(
        layer_size: u32,
        activation_function: fn(f32) -> f32,
        activation_derivation: fn(f32) -> f32,
        weight_function: fn(u32) -> Array1<f32>,
        prev_layer: &dyn Layer,
    ) -> Self {
        let weights = weight_function(prev_layer.weights_size());
        let mut outputs: Vec<Neuron> = vec![
            Neuron::Output(Output {
                input_value: 0.0,
                output_value: 0.0,
                weights
            });
            layer_size.try_into().unwrap()
        ];
        Self {
            outputs : Array1::from_vec(outputs),
            activation_function,
            activation_derivation,
        }
    }
}

impl Layer for OutputLayer {
    fn weights_size(&self) -> u32 {
        self.outputs.len().try_into().unwrap()
    }

    fn values_as_arr(&self) -> Array1<f32> {
        Array1::from_vec(
            self.outputs
                .iter()
                .map(|n| match n {
                    Neuron::Input(i) => i.input_value,
                    _ => 0.0,
                })
                .collect(),
        )
    }

    fn len(&self) -> usize {
        self.outputs.len()
    }


    fn get(&self, index: usize) -> Option<&Neuron> {
        self.outputs.get(index)
    }

    fn get_mut(&mut self, index: usize) -> Option<&mut Neuron> {
        self.outputs.get_mut(index)
    }

    fn get_all(&self) -> &Array1<Neuron> {
        &self.outputs
    }


    fn get_all_mut(&mut self) -> &mut Array1<Neuron> {
        &mut self.outputs
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_activation_derivation(&self) -> fn(f32) -> f32 {
       self.activation_derivation
    }
}

impl fmt::Display for OutputLayer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "neurons: {:?}\n", self.outputs)
    }
}

#[derive(Debug, Clone)]
pub struct HiddenLayer {
    pub neurons: Array1<Neuron>,
    pub activation_function: fn(f32) -> f32,
    pub activation_derivation: fn(f32) -> f32,
    pub bias: bool,
}

impl HiddenLayer {
    pub fn new(
        layer_size: u32,
        bias: bool,
        activation_function: fn(f32) -> f32,
        activation_derivation: fn(f32) -> f32,
        weight_function: fn(u32) -> Array1<f32>,
        prev_layer: &dyn Layer,
    ) -> Self {
        let weights = weight_function(prev_layer.weights_size());
        let mut neurons: Vec<Neuron> = vec![
            Neuron::Hidden(Hidden {
                input_value: 0.0,
                output_value: 0.0,
                weights
            });
            layer_size.try_into().unwrap()
        ];
        if bias {
            neurons.push(Neuron::Bias(Bias { input_value: 1.0, output_value: 1.0 }));
        }
        Self {
            neurons : Array1::from_vec(neurons),
            activation_function,
            activation_derivation,
            bias,
        }
    }
}

impl Layer for HiddenLayer{
    fn weights_size(&self) -> u32 {
        self.neurons[0].len().try_into().unwrap()
    }

    fn values_as_arr(&self) -> Array1<f32> {
        Array1::from_vec(
            self.neurons
                .iter()
                .map(|n| match n {
                    Neuron::Hidden(i) => i.input_value,
                    _ => 0.0,
                })
                .collect(),
        )
    }

    fn len(&self) -> usize {
        self.neurons.len()
    }

    fn get(&self, index: usize) -> Option<&Neuron> {
        self.neurons.get(index)
    }

    fn get_mut(&mut self, index: usize) -> Option<&mut Neuron> {
        self.neurons.get_mut(index)
    }

    fn get_all(&self) -> &Array1<Neuron> {
        &self.neurons
    }

    fn get_all_mut(&mut self) -> &mut Array1<Neuron> {
        &mut self.neurons
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_activation_derivation(&self) -> fn(f32) -> f32 {
        self.activation_derivation
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
    use crate::net::activation_functions::sigmoid_derivative;

    use super::*;

    #[test]
    fn layer_test() {
        let input = InputLayer::new(2, true);
        let a = HiddenLayer::new(5, true, sigmoid, sigmoid_derivative, xavier_init, &input);
        println!("{:?}", a);
        assert_eq!(1, 0);
    }
}
