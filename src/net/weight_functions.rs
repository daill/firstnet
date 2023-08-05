use rand::{thread_rng, Rng};

pub fn xavier_init(size: u32) -> Vec<f32> {
    let lower = -(1.0 / f32::sqrt(size as f32));
    let upper = -lower;
    vec![thread_rng().gen_range(lower..upper)]
}

pub fn zero_value_init(size: u32) -> Vec<f32> {
    vec![0.0; size as usize]
}
