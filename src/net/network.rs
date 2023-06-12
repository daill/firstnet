use crate::net::layer::Layer;

#[derive(Debug, Clone)]
pub struct Network {
    input_layer: Layer,
    hidden_layer: Vec<Layer>,
    output_layer: Layer,
}

impl Network {
    fn new(input_layer: Layer, hidden_layer: Vec<Layer>, output_layer: Layer) -> Network {
        Network {
            input_layer,
            hidden_layer,
            output_layer,
        }
    }
}
