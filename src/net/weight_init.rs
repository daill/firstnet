use rand::Rng;

use super::{neuron::Neuron, node::Neuron};

pub fn xavier_init(num_inputs: u32, size: u32) -> Vec<Neuron> {
    let lower = -(1.0 / f32::sqrt(num_inputs as f32));
    let upper = -lower;
    let mut neurons: Vec<Neuron> = Vec::with_capacity(size.try_into().unwrap());
    for n in 1..size {
        neurons.push(Neuron::from_weight(
            rand::thread_rng().gen_range(lower..upper),
        ));
    }

    return neurons;
}

pub fn zero_init(size: u32) -> Vec<f32> {
    let neurons: Vec<Neuron> = Vec::with_capacity(size.try_into().unwrap());
    for n in 1..size {
        neurons.push(Neuron::from_value(0.0));
    }
}
