use std::fmt;

#[derive(Clone, Debug)]
pub struct Neuron {
    value: f32,
    weights: Vec<f32>,
}

impl Neuron {
    pub fn new(weights_size: u32) -> Neuron {
        Neuron {
            value: 0.0,
            weights: vec![0.0; weights_size as usize],
        }
    }

    pub fn from_value(value: f32, weights_size: u32) -> Neuron {
        Neuron {
            value,
            weights: vec![0.0; weights_size as usize],
        }
    }

    pub fn from_weights(weights: Vec<f32>) -> Neuron {
        Neuron {
            value: 0.0,
            weights,
        }
    }

    pub fn init(&mut self, f: fn(u32) -> Vec<f32>) {
        self.weights = f(self.weights.len().try_into().unwrap());
    }
}

impl fmt::Display for Neuron {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "value: {}", self.value)
    }
}
