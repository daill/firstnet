use rand::{thread_rng, Rng};

use super::neuron::Neuron;

pub fn xavier_init(num_inputs: u32, size: u32) -> Vec<Neuron> {
    let lower = -(1.0 / f32::sqrt(num_inputs as f32));
    let upper = -lower;
    let mut neurons: Vec<Neuron> = Vec::with_capacity(size.try_into().unwrap());
    for _n in 0..size {
        neurons.push(Neuron::from_weights(vec![
            thread_rng().gen_range(lower..upper)
        ]));
    }

    neurons
}

pub fn zero_value_init(size: u32) -> Vec<Neuron> {
    let mut neurons: Vec<Neuron> = Vec::with_capacity(size.try_into().unwrap());
    for _n in 0..size {
        neurons.push(Neuron::from_value(0.0));
    }
    neurons
}
