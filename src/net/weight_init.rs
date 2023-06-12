use rand::Rng;

use super::node::Node;

pub fn xavier_init(num_inputs: u32, size: u32) -> Vec<Node> {
    let lower = -(1.0 / f32::sqrt(num_inputs as f32));
    let upper = -lower;
    let mut nodes: Vec<Node> = Vec::with_capacity(size.try_into().unwrap());
    for n in 1..size {
        nodes.push(Node::from_weight(
            rand::thread_rng().gen_range(lower..upper),
        ));
    }

    return nodes;
}

pub fn zero_init(size: u32) -> Vec<f32> {
    vec![0.0]
}
