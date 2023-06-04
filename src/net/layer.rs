struct Layer {
    weights: Vec<f32>, 
    bias: f32,
}

impl Layer {
    fn new(input_size: u32) -> Layer{ 
        Layer {
            bias: 1.2, 
            weights: vec![0],
        }
    }
}
