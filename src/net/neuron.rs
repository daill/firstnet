#[derive(Clone, Debug)]
pub struct Neuron {
    value: f32,
    weight: f32,
}

impl Neuron {
    pub fn new() -> Neuron {
        Neuron {
            value: 1.2,
            weight: 0.0,
        }
    }

    pub fn from_weight(weight: f32) -> Neuron {
        Neuron { value: 0.0, weight }
    }

    pub fn from_value(value: f32) -> Neuron {
        Neuron {
            value,
            weight: -2.0f32,
        }
    }
}
