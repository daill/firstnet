use ndarray::prelude::*;
use rand::{thread_rng, Rng};

pub fn xavier_init(size: u32) -> Array1<f32> {
    let lower = -(1.0 / f32::sqrt(size as f32));
    let upper = -lower;
    let data: Vec<f32> = (0..size)
        .map(|_| thread_rng().gen_range(lower..upper))
        .collect();
    Array::from_vec(data)
}
