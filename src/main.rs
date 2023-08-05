use crate::net::{
    layer::{HiddenLayer, InputLayer, OutputLayer},
    network::Network,
};

mod net;

fn main() {
    println!("Hello, world!");
    let input_layer = InputLayer::new(2);
    let output_layer = OutputLayer::new(2, 2);
    let hidden_a = HiddenLayer::new(2, 2);
    let hidden_b = HiddenLayer::new(5, 2);
    let network = Network::new(input_layer, vec![hidden_a, hidden_b], output_layer);
    print!("{}", network);
    network.feed_forward();
}
