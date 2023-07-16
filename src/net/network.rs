use std::fmt;

use crate::net::layer::Layer;

#[derive(Debug, Clone)]
pub struct Network {
    input_layer: Layer,
    hidden_layer: Vec<Layer>,
    output_layer: Layer,
}

impl Network {
    pub fn new(input_layer: Layer, hidden_layer: Vec<Layer>, output_layer: Layer) -> Self {
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
            }
            println!("test");
            print!("{:?}", layer);
        }
    }

    pub fn calc_values(l1: &mut Layer, l2: &mut Layer) {
        for (i, target_neuron) in l2.0.iter().enumerate() {
            for (j, source_neuron) in l1.0.iter().enumerate() {}
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
