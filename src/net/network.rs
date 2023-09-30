use std::fmt;

use crate::net::layer::Layer;
use itertools::Itertools;

use super::{
    layer::{HiddenLayer, InputLayer, OutputLayer},
    neuron::{Bias, Hidden, Neuron},
};

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
        let inputs = input.get_values_as_arr();
        // feed the first layer
        {
            let first_layer = self.hidden_layer.get_mut(0).unwrap();
            for n in 0..first_layer.neurons.len() {
                let neuron = first_layer.neurons.get_mut(n).unwrap();
                match neuron {
                    Neuron::Hidden(h) => {
                        let cal_val = inputs.dot(&h.weights);
                        h.value = (first_layer.activation_function)(cal_val);
                    }
                    _ => {}
                };
            }
        }

        {
            let hidden_layers = &mut self.hidden_layer;
            if (hidden_layers).len() > 1 {
                let iter = (0..hidden_layers.len() - 1).into_iter();
                for (prev, next) in iter.tuple_windows() {
                    let prev_layer = hidden_layers[prev];
                    let next_layer = &mut hidden_layers[next];
                    let values = prev_layer.get_values_as_arr();
                    for n in 0..next_layer.neurons.len() {
                        let neuron = next_layer.neurons.get_mut(n).unwrap();
                        match neuron {
                            Neuron::Hidden(h) => {
                                let cal_val = values.dot(&h.weights);
                                //h.value = (hidden_layers.get(0).unwrap().activation_function)(cal_val);
                            }
                            _ => {}
                        }
                    }
                }
            }
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
        writeln!(
            f,
            "input: {:?}\nhidden: {:?}\noutput: {:?}",
            self.input_layer, self.hidden_layer, self.output_layer
        )
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use crate::net::{activation_functions, weight_functions::xavier_init};

    use super::*;

    #[test]
    fn network_feed_forward_test() {
        let mut input_layer = InputLayer::new(2, true);
        input_layer.set_inputs(vec![1.0, 1.0]);
        let mut hidden_a = HiddenLayer::new(
            5,
            true,
            activation_functions::sigmoid,
            xavier_init,
            &input_layer,
        );
        let mut output = OutputLayer::new(2, activation_functions::sigmoid, xavier_init, &hidden_a);
        let mut net = Network::new(input_layer, vec![hidden_a], output);
        net.feed_forward();
        println!("{:?}", &net.hidden_layer[0]);
        assert_eq!(1, 0);
    }
}
