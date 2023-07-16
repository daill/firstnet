use std::fmt;

#[derive(Clone, Debug)]
pub struct Neuron(f32, Vec<f32>);

impl Neuron {
    pub fn new() -> Neuron {
        Neuron(1.2, vec![0.0])
    }

    pub fn from_weights(weights: Vec<f32>) -> Neuron {
        Neuron(0.0, weights)
    }

    pub fn from_value(value: f32) -> Neuron {
        Neuron(value, vec![-2.0f32])
    }
}

impl fmt::Display for Neuron {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "value: {} weight: {:?}", self.0, self.1)
    }
}
