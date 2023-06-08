use crate::net::node::Node;
use crate::net::weight_init;

use self::weight_init::he_init;

struct Layer {
    weights: Vec<Node>,
    bias: f32,
}

impl Layer {
    fn new(input_size: u32) -> Layer {
        he_init(5);
        Layer {
            bias: 1.2,
            weights: vec![Node],
        }
    }
}
