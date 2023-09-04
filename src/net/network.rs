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
        self.pre_flight_check();
        // for every value inside the input layer
        // get each hidden layer and calculate

        let input = &self.input_layer;
        let mut layer = self.hidden_layer.get_mut(0).unwrap();
        // feed the first layer
        for n in 0..layer.neurons.len() {
            let mut neuron = layer.neurons.get_mut(n).unwrap();
            let cal_val = input.inputs.dot(&neuron.weights);
            neuron.value = (layer.activation_function)(cal_val);
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

    fn pre_flight_check(self) {
        if self.input_layer.inputs.len()
            != self.hidden_layer[0].neurons.first().unwrap().weights.len()
        {
            panic!("invalid input layer size");
        }
        for n in 1..self.hidden_layer.len() {
            if self.hidden_layer[n].neurons.first().unwrap().weights.len()
                != self.hidden_layer[n - 1].neurons.len()
            {
                panic!("invalid hidden layer size");
            }
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
        let mut input_layer = InputLayer::new(2);
        input_layer.inputs = array![1.0, 1.0];
        let mut hidden_a = HiddenLayer::new(5, 2, activation_functions::sigmoid);
        hidden_a.init(xavier_init);
        let mut net = Network::new(
            input_layer,
            vec![hidden_a],
            OutputLayer::new(2, 2, activation_functions::sigmoid),
        );
        net.feed_forward();
        println!("{:?}", &net.hidden_layer[0]);
        assert_eq!(1, 0);
    }
}
