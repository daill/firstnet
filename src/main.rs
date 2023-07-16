use crate::net::{layer::Layer, network::Network};

mod net;

fn main() {
    println!("Hello, world!");
    let input_layer = Layer::new_inputs(2);
    let output_layer = Layer::new_outputs(1);
    let hidden_a = Layer::new(1, 5);
    let hidden_b = Layer::new(5, 5);
    let network = Network::new(input_layer, vec![hidden_a, hidden_b], output_layer);
    print!("{}", network);
    network.feed_forward();
}
