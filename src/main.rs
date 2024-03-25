use ndarray::array;

use crate::net::{
    activation_functions::{self, sigmoid},
    layer::{HiddenLayer, InputLayer, OutputLayer},
    network::Network,
    weight_functions::xavier_init,
};

mod net;

fn main() {
    println!("Hello, world!");

}
