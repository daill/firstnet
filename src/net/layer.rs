use crate::net::node::Neuron;

use super::weight_init::xavier_init;

#[derive(Debug, Clone)]
pub struct Layer {
    nodes: Vec<Neuron>,
}

impl Layer {
    pub fn new(input_size: u32) -> Layer {
        let nodes: Vec<Neuron> = xavier_init(5, 5);
        Layer { nodes }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_test() {
        let a = Layer::new(5);
        println!("{:?}", a);
        assert_eq!(1, 0);
    }
}
