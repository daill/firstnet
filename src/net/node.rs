#[derive(Clone, Debug)]
pub struct Node {
    bias: f32,
    weight: f32,
}

impl Node {
    pub fn new() -> Node {
        Node {
            bias: 1.2,
            weight: 0.0,
        }
    }

    pub fn from_weight(weight: f32) -> Node {
        Node { bias: 1.2, weight }
    }
}
