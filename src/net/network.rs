use std::fmt;

use crate::net::layer::Layer;
use itertools::Itertools;
use ndarray::{Array0, Array1, ArrayBase};

use super::{
    activation_functions,
    layer::{HiddenLayer, InputLayer, OutputLayer},
    neuron::{Bias, Hidden, Neuron},
    weight_functions::xavier_init,
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
        let first_layer = self.hidden_layer.get_mut(0).unwrap();
        for n in 0..first_layer.neurons.len() {
            let mut neuron = first_layer.neurons.get_mut(n).unwrap();
            if let Neuron::Hidden(h) = neuron {
                let cal_val = inputs.dot(&h.weights);
                h.value = (first_layer.activation_function)(cal_val);
            };
        }

        let hidden_layers = &mut self.hidden_layer;
        if (hidden_layers).len() > 1 {
            let iter = (0..hidden_layers.len() - 1).into_iter();
            for (prev, next) in iter.tuple_windows() {
                let prev_layer = hidden_layers.get(prev).unwrap();
                let values = prev_layer.get_values_as_arr();
                let next_layer = &mut hidden_layers[next];
                for n in 0..next_layer.neurons.len() {
                    let mut neuron = next_layer.neurons.get_mut(n).unwrap();
                    if let Neuron::Hidden(h) = neuron {
                        let cal_val = values.dot(&h.weights);
                        h.value = (next_layer.activation_function)(cal_val);
                    }
                }
            }
        }

        let output_layer = &mut self.output_layer;
        let last_hidden = hidden_layers.last().unwrap();
        let layer_values = last_hidden.get_values_as_arr();
        for n in 0..output_layer.outputs.len() {
            let neuron = output_layer.outputs.get_mut(n).unwrap();

            if let Neuron::Output(o) = neuron {
                let cal_val = layer_values.dot(&o.weights);
                o.value = (output_layer.activation_function)(cal_val);
            }
        }

        println!("{:?}", self.input_layer);
        for i in 1..self.hidden_layer.len() {
            println!("{:?}", self.hidden_layer[i]);
        }
        println!("{:?}", output_layer);
    }

    pub fn backward_pass(&mut self) {
        let mut output_layer = &mut self.output_layer;
        let last_hidden = self.hidden_layer.last_mut().unwrap();
        for n in 0..output_layer.outputs.len() {
            let mut outputs = &mut output_layer.outputs;
            let output_neuron = outputs.get_mut(n).unwrap();
            if let Neuron::Output(neuron) = output_neuron {
                let neuron_weights = &mut neuron.weights;
                for i in 0..neuron_weights.len() {
                    let hidden_neuron = last_hidden.neurons.get_mut(i).unwrap();
                    let nn = match hidden_neuron {
                        Neuron::Hidden(h) => h,
                        _ => panic!(),
                    };

                    neuron_weights[i] = neuron_weights.get(i).unwrap() - 0.05 * (-0.809) * nn.value;
                }
            }
        }

        for i in (self.hidden_layer.len() - 1)..1 {
            let mut hidden_layer = self.hidden_layer.get_mut(i).unwrap();
            for n in 0..hidden_layer.neurons.len() {
                let mut neuron = hidden_layer.neurons.get_mut(n).unwrap();
                if let Neuron::Hidden(h) = neuron {
                    let neuron_weights = &mut h.weights;
                    for i in 0..neuron_weights.len() {
                        let input_neuron = self.input_layer.neurons.get(i).unwrap();
                    }
                }
            }
        }
    }

    pub fn calc_new_weights(&mut self) {}

    pub fn calc_total_error(&mut self, values: Array1<f32>, expected: Array1<f32>) -> f32 {
        let mut error = 0.0;
        for n in 0..values.len() {
            error += 0.5 * (values.get(n).unwrap() - expected.get(n).unwrap()).powf(2.0);
        }

        return error;
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

    use crate::net::{activation_functions, neuron::Output, weight_functions::xavier_init};

    use super::*;

    fn setup() -> Network {
        let mut input_layer = InputLayer::new(2, true);
        input_layer.set_inputs(vec![2.0, 3.0]);
        let mut hidden_a = HiddenLayer::new(
            2,
            true,
            activation_functions::nop,
            xavier_init,
            &input_layer,
        );
        hidden_a.neurons = vec![
            Neuron::Hidden(Hidden {
                value: 0.0,
                weights: array![0.11, 0.21],
            }),
            Neuron::Hidden(Hidden {
                value: 0.0,
                weights: array![0.12, 0.08],
            }),
        ];
        let mut output = OutputLayer::new(1, activation_functions::nop, xavier_init, &hidden_a);
        output.outputs = vec![Neuron::Output(Output {
            value: 0.0,
            weights: array![0.14, 0.15],
        })];

        Network::new(input_layer, vec![hidden_a], output)
    }

    #[test]
    fn network_feed_forward_test() {
        let mut net = setup();
        net.feed_forward();

        println!("{:?}", &net.input_layer);
        println!("{:?}", &net.hidden_layer[0]);
        println!("{:?}", &net.output_layer);
        assert_eq!(1, 0);
    }

    #[test]
    fn calc_error_test() {
        let mut net = setup();
        let values = array![0.191];
        let expected = array![1.0];
        assert_eq!(0.32724053, net.calc_total_error(values, expected));
    }

    #[test]
    fn network_backward_pass_test() {
        let mut net = setup();
        net.feed_forward();

        net.backward_pass();

        println!("{:?}", &net);
        assert_eq!(1, 0);
    }
}
