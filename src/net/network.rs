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

    pub fn feed_forward(&self) {
        // for every value inside the input layer
        // get each hidden layer and calculate
        let mut layers = self.hidden_layer.iter();

        for (i, layer) in layers.enumerate() {
            if i == 0 {
                // feed the first layer
                for &mut neuron in layer.neurons.iter_mut() {}
            }
            println!("test");
            print!("{:?}", layer);
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
