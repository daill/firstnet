use std::any::{Any, TypeId};
use std::fmt;
use std::fmt::Debug;

use crate::net::layer::Layer;
use itertools::Itertools;
use ndarray::{Array0, Array1, ArrayBase};
use crate::net::neuron::NeuronBase;

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

    pub fn forward_pass(&mut self) {
        // for every value inside the input layer
        // get each hidden layer and calculate

        let input = &self.input_layer;
        let inputs = input.values_as_arr();
        // feed the first layer
        let first_layer = self.hidden_layer.get_mut(0).unwrap();
        for n in 0..first_layer.neurons.len() {
            let mut neuron = first_layer.neurons.get_mut(n).unwrap();
            if let Neuron::Hidden(h) = neuron {
                h.input_value = inputs.dot(&h.weights);
                h.output_value = (first_layer.activation_function)(h.input_value);
            };
        }

        let hidden_layers = &mut self.hidden_layer;
        if (hidden_layers).len() > 1 {
            let iter = (1..hidden_layers.len() - 1).into_iter();
            for (prev, next) in iter.tuple_windows() {
                let prev_layer = hidden_layers.get(prev).unwrap();
                let values = prev_layer.values_as_arr();
                let next_layer = &mut hidden_layers[next];
                for n in 0..next_layer.neurons.len() {
                    let mut neuron = next_layer.neurons.get_mut(n).unwrap();
                    if let Neuron::Hidden(h) = neuron {
                        h.input_value = values.dot(&h.weights);
                        h.output_value = (next_layer.activation_function)(h.input_value);
                    }
                }
            }
        }

        let mut output_layer = &mut self.output_layer;
        let last_hidden = hidden_layers.last().unwrap();
        let layer_values = last_hidden.values_as_arr();
        for n in 0..output_layer.outputs.len() {
            let neuron = output_layer.outputs.get_mut(n).unwrap();

            if let Neuron::Output(o) = neuron {
                o.input_value = layer_values.dot(&o.weights);
                o.output_value = (output_layer.activation_function)(o.input_value);
            }
        }
    }




    pub fn backward_pass(&mut self, expected: Vec<f32>) -> f32 {
        //
        let mut global_error = 0.0;
        let learning_rate = 0.03;
        let mut output_layer: &mut dyn Layer = &mut self.output_layer;
        let mut neuron_deltas: Vec<f32> = vec![0.0; self.hidden_layer[self.hidden_layer.len()-1].len()];
        for n in 0..output_layer.len() {
            let mut outputs = &mut output_layer.get_all_mut();
            let neuron = outputs.get_mut(n).unwrap();
            let delta = neuron.get_output_value() - expected[n];
            global_error += 0.5 * delta.powf(2.0);

            let neuron_weights = neuron.get_mut_weights();
            for i in 0..neuron_weights.len() {
                neuron_deltas[i] += delta * neuron_weights[i];
            }
        }

        let mut hidden_layer:  &mut HiddenLayer = self.hidden_layer.last_mut().unwrap();
        let activation_derivation = hidden_layer.activation_derivation;
        for ln in 0..output_layer.len() {
            // calc new weights
            let mut ll_neuron: &mut Neuron = output_layer.get_mut(ln).unwrap();
            let mut output_value = 0.0;
            let mut ll_weights = ll_neuron.get_mut_weights();
            for lw in 0..ll_weights.len() {
                let mut hn_out = hidden_layer.get_mut(lw).unwrap();
                output_value = hn_out.get_output_value();
                ll_weights[lw] -= learning_rate * activation_derivation(output_value) * neuron_deltas[lw];
            }

        }


        // let iter = (0..self.hidden_layer.len()).rev().into_iter();
        //
        // let mut hidden_layer = &mut self.hidden_layer;
        // for (prev, next) in iter.tuple_windows() {
        //
        //
        //     let next_layer = hidden_layer.get(next).unwrap();
        //     let next_layer_values = next_layer.values_as_arr();
        //     let mut current_layer = &mut hidden_layer[prev];
        //
        //     let mut temp_deltas = vec![0.0; current_layer.len()];
        //
        //     let activation_derivation = current_layer.activation_derivation;
        //     for n in 0..current_layer.neurons.len() {
        //         let mut neuron = current_layer.neurons.get_mut(n).unwrap();
        //
        //         if let Neuron::Hidden(h) = neuron {
        //             let neuron_weights = &mut h.weights;
        //             for i in 0..neuron_weights.len() {
        //                 temp_deltas[i] += neuron_weights[i] * neuron_deltas[i];
        //             }
        //         }
        //     }
        //     for ln in 0..current_layer.len() {
        //         // calc new weights
        //         let mut ll_neuron: &mut dyn NeuronBase = current_layer.get_mut(ln).unwrap();
        //         let mut output_value = 0.0;
        //         let mut ll_weights = ll_neuron.get_mut_weights();
        //         for lw in 0..ll_weights.len() {
        //             output_value = next_layer_values[lw];
        //             ll_weights[lw] -= learning_rate * activation_derivation(output_value) * temp_deltas[lw];
        //         }
        //
        //     }
        //
        //     neuron_deltas = temp_deltas;
        // }

        let mut next_layer = &mut self.input_layer;
        let next_layer_values = next_layer.values_as_arr();
        let mut current_layer = &mut self.hidden_layer[0];

        let mut temp_deltas = vec![0.0; current_layer.len()];

        let activation_derivation = current_layer.activation_derivation;
        for n in 0..current_layer.neurons.len() {
            let mut neuron = current_layer.neurons.get_mut(n).unwrap();

            if let Neuron::Hidden(h) = neuron {
                let neuron_weights = &mut h.weights;
                for i in 0..neuron_weights.len() {
                    temp_deltas[i] += neuron_weights[i] * neuron_deltas[i];
                }
            }
        }
        for ln in 0..current_layer.len() {
            // calc new weights
            let mut ll_neuron: &mut Neuron = current_layer.get_mut(ln).unwrap();
            let mut output_value = 0.0;
            let mut ll_weights = ll_neuron.get_mut_weights();
            for lw in 0..ll_weights.len() {
                output_value = next_layer_values[lw];
                ll_weights[lw] -= learning_rate * activation_derivation(output_value) * temp_deltas[lw];

            }

        }

        return global_error;
    }


    pub fn calc_total_error(&mut self, expected: Array1<f32>) -> f32 {
        let mut error = 0.0;
        for n in 0..self.output_layer.len() {
            error += 0.5 * (self.output_layer.get(n).unwrap().get_output_value() - expected.get(n).unwrap()).powf(2.0);
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
    use rand::Rng;

    use crate::net::{activation_functions, neuron::Output, weight_functions::xavier_init};

    use super::*;

    fn setup() -> Network{
        let mut input_layer = InputLayer::new(2, true);
        let mut hidden_a = HiddenLayer::new(
            3,
            true,
            activation_functions::sigmoid,
            activation_functions::sigmoid_derivative,
            xavier_init,
            &input_layer,
        );
        let neurons = vec![
            Neuron::Hidden(Hidden {
                input_value: 0.0,
                weights: array![0.11, 0.21, 0.31],
                output_value: 0.0,
            }),
            Neuron::Hidden(Hidden {
                input_value: 0.0,
                weights: array![0.12, 0.08, 0.04],
                output_value: 0.0,
            }),

            Neuron::Bias(Bias {
                input_value: 1.0,
                output_value: 1.0,
                weights: array![0.0],
            })
        ];
        hidden_a.neurons = Array1::from_vec(neurons);

        let neurons_b = vec![
            Neuron::Hidden(Hidden {
                input_value: 0.0,
                weights: array![0.11, 0.21, 0.31],
                output_value: 0.0,
            }),
            Neuron::Hidden(Hidden {
                input_value: 0.0,
                weights: array![0.12, 0.08, 0.04],
                output_value: 0.0,
            }),

            Neuron::Bias(Bias {
                input_value: 1.0,
                output_value: 1.0,
                weights: array![0.0],
            })
        ];

        let mut hidden_ab = HiddenLayer::new(
            2,
            true,
            activation_functions::nop,
            activation_functions::nop_derivative,
            xavier_init,
            &hidden_a,
        );
        hidden_ab.neurons = Array1::from_vec(neurons_b);
        let mut output = OutputLayer::new(1, activation_functions::nop, activation_functions::nop_derivative, xavier_init, &hidden_a);
        let outputs = vec![Neuron::Output(Output {
            input_value: 0.0,
            weights: array![0.14, 0.15, 0.16],
            output_value: 0.0,
        })];
        output.outputs = Array1::from_vec(outputs);

        Network::new(input_layer, vec![hidden_a ], output)
    }

    #[test]
    fn network_feed_forward_test() {
        let mut net = setup();
        net.input_layer.set_inputs(vec![2.0, 3.0]);
        net.forward_pass();

        assert_eq!(1, 0);
    }

    #[test]
    fn calc_error_test() {
        let mut net = setup();
        let values = array![0.191];
        let expected = array![1.0];
        assert_eq!(0.32724053, net.calc_total_error(expected));
    }

    #[test]
    fn network_backward_pass_test() {
        let mut net = setup();
        net.input_layer.set_inputs(vec![1.0, 1.0]);
        net.forward_pass();
        println!("{:?}", &net);
        net.backward_pass(vec![1.0]);


        println!("{:?}", &net);
        assert_eq!(1, 0);
    }

    #[test]
    fn network_training_test() {
        let mut net = setup();

        let mut error = 1.0;
        let data = array![[0.0, 0.0, 0.0],
                                        [1.0, 0.0, 1.0],
                                        [1.0, 1.0, 0.0],
                                        [0.0, 1.0, 1.0]];

        let iterations = 100;
        let mut rng = rand::thread_rng();

        let mut error_vec: Vec<f32> = vec![];

        println!("{:?}", &net);
        for i in 0..iterations {
            let index = rng.gen_range(0..data.shape()[0]-1);

            net.input_layer.set_inputs(vec![data[[index,1]], data[[index,2]]]);
            net.forward_pass();
            error = net.backward_pass(vec![data[[index, 0]]]);

            if i % 10 == 0 {
                let mut total_error = 0.0;
                for n in 0..(data.shape()[0]) {
                    net.input_layer.set_inputs(vec![data[[n,1]], data[[n,2]]]);
                    net.forward_pass();

                    total_error =net.output_layer.get(0).unwrap().get_output_value();
                    // total_error += 0.5 * (net.output_layer.get(0).unwrap().get_output_value() - data[[n, 0]]).powf(2.0);
                    println!("{}: {} - {} {}", i, total_error, data[[n,1]], data[[n,2]]);
                }
                println!("{}: {}", i, total_error);
                println!("{:?}", &net);
            }

        }

        println!("{:?}", &net);
        net.input_layer.set_inputs(vec![1.0, 1.0]);
        net.forward_pass();
        println!("{:?}", net.output_layer.get_all().get(0).unwrap().get_output_value());

        net.input_layer.set_inputs(vec![0.0, 1.0]);
        net.forward_pass();
        println!("{:?}", net.output_layer.get_all().get(0).unwrap().get_output_value());


    }
}
