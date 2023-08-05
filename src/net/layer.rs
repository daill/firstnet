use std::fmt;

use crate::net::neuron::Neuron;

pub(crate) trait Layer {}

#[derive(Debug, Clone)]
pub struct InputLayer {
    pub inputs: Vec<f32>,
}

impl Layer for InputLayer {}

impl InputLayer {
    fn from_inputs(inputs: Vec<f32>) -> Self {
        Self { inputs }
    }

    pub fn new(layer_size: u32) -> Self {
        Self {
            inputs: vec![0.0; layer_size as usize],
        }
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
}

impl OutputLayer {
    pub fn new(layer_size: u32, weights_size: u32) -> Self {
        Self {
            outputs: vec![Neuron::new(weights_size); layer_size.try_into().unwrap()],
        }
    }
}

impl fmt::Display for OutputLayer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "neurons: {:?}\n", self.outputs)
    }
}

#[derive(Debug, Clone)]
pub struct HiddenLayer {
    pub neurons: Vec<Neuron>,
}

impl HiddenLayer {
    pub fn new(layer_size: u32, weights_size: u32) -> Self {
        let neurons: Vec<Neuron> = vec![Neuron::new(weights_size); layer_size.try_into().unwrap()];
        Self { neurons }
    }
}

impl HiddenLayer {
    fn init(&mut self, f: fn(u32) -> Vec<f32>) {
        self.neurons.iter_mut().for_each(|neuron| {
            neuron.init(f);
        });
    }
}

impl fmt::Display for HiddenLayer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "neurons: {:?}\n", self.neurons)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_test() {
        let a = HiddenLayer::new(5, 5);
        println!("{:?}", a);
        assert_eq!(1, 0);
    }
}
