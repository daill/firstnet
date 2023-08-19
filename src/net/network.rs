use std::fmt;

use crate::net::layer::Layer;

use super::layer::{HiddenLayer, InputLayer, OutputLayer};

#[derive(Debug, Clone)]
pub struct Network {
    input_layer: InputLayer,
    hidden_layer: Vec<HiddenLayer>,
    output_layer: OutputLayer,
}

impl Network {
    pub fn new(
        input_layer: InputLayer,
        hidden_layer: Vec<HiddenLayer>,
        output_layer: OutputLayer,
    ) -> Self {
        Self {
            input_layer,
            hidden_layer,
            output_layer,
        }
    }

    pub fn feed_forward(&mut self) {
        // for every value inside the input layer
        // get each hidden layer and calculate

        let input = &self.input_layer;
        let mut layer = self.hidden_layer.get_mut(0).unwrap();
        // feed the first layer
        for n in 0..layer.neurons.len() {
            let mut neuron = layer.neurons.get_mut(n).unwrap();
            neuron.value = input.inputs.dot(&neuron.weights);
        }

        for i in 1..self.hidden_layer.len() {
            println!("test");
            print!("{:?}", self.hidden_layer[i]);
        }
    }

    pub fn calc_values(l1: &mut HiddenLayer, l2: &mut HiddenLayer) {
        for (i, target_neuron) in l2.neurons.iter().enumerate() {
            for (j, source_neuron) in l1.neurons.iter().enumerate() {}
        }
    }
}

impl fmt::Display for Network {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{:?}\n{:?}\n{:?}",
            self.input_layer, self.hidden_layer, self.output_layer
        )
    }
}
