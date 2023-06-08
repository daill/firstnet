pub use crate::net::node::Node;

struct Layer {
    weights: Vec<Node>,
    bias: f32,
}

impl Layer {
    fn new(input_size: u32) -> Layer {
        Layer {
            bias: 1.2,
            weights: vec![0f32],
        }
    }
}
