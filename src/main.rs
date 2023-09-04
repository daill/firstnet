use ndarray::array;

use crate::net::{
    activation_functions::{self, sigmoid},
    layer::{HiddenLayer, InputLayer, OutputLayer},
    network::Network,
    weights::xavier_init,
};

mod net;

fn main() {
    println!("Hello, world!");
    let mut input_layer = InputLayer::new(2, true);

    let hidden_a = HiddenLayer::new(2, true, sigmoid, xavier_init, &input_layer);
    let hidden_b = HiddenLayer::new(5, true, sigmoid, xavier_init, &hidden_a);

    let output_layer = OutputLayer::new(2, 2, sigmoid);
    let mut network = Network::new(input_layer, vec![hidden_a, hidden_b], output_layer);
    print!("{}", network);
    network.feed_forward();
}
