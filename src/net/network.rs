use crate::net::layer::Layer;

#[derive(Debug, Clone)]
pub struct Network {
    input_layer: Layer,
    hidden_layer: Vec<Layer>,
    output_layer: Layer,
}

impl Network {
    pub fn new(input_layer: Layer, hidden_layer: Vec<Layer>, output_layer: Layer) -> Self {
        Self {
            input_layer,
            hidden_layer,
            output_layer,
        }
    }

    pub fn run() {
        // for every value inside the input layer 
        // get each hidden layer
    }
}
