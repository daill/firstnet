#[derive(Clone, Debug)]
pub struct Neuron {
    bias: f32,
    weight: f32,
}

impl Neuron {
    pub fn new() -> Neuron {
        Neuron {
            bias: 1.2,
            weight: 0.0,
        }
    }

    pub fn from_weight(weight: f32) -> Neuron {
        Neuron { bias: 1.2, weight }
    }
}
