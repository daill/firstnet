use crate::net::weight_functions::zero_value_init;
use std::fmt;

use crate::net::neuron::Neuron;

#[derive(Debug, Clone)]
pub struct HiddenLayer {
    nodes: Vec<Neuron>,
    layer_size: i32,
}

impl HiddenLayer {
    pub fn new(layer_size: u32) -> Self {
        let nodes: Vec<Neuron> = zero_value_init(layer_size);
        let layer_size = nodes.len();
        HiddenLayer { nodes, layer_size }
    }

    // returns the weight array of each node in an array
    pub fn get_tensor(&self) -> [[f32]] {
        return [[0.0; self.0.first().unwrap().len()]; self.0.len()];
    }
}

impl fmt::Display for InputLayer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "nodes: {:?}\n", self.0)
    }
}

#[derive(Debug, Clone)]
pub struct InputLayer(pub Vec<Neuron>);

impl InputLayer {
    pub fn new(layer_size: u32) -> Self {
        let nodes: Vec<Neuron> = zero_value_init(layer_size);
        InputLayer(nodes)
    }
}

#[derive(Debug, Clone)]
pub struct OutputLayer(pub Vec<Neuron>);

impl OutputLayer {
    pub fn new(layer_size: u32) -> Self {
        let nodes: Vec<Neuron> = zero_value_init(layer_size);
        OutputLayer(nodes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_test() {
        let a = Layer::new(5, 5);
        println!("{:?}", a);
        assert_eq!(1, 0);
    }
}
