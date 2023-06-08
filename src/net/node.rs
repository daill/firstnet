pub struct Node {
    bias: f32,
    weight: f32,
}


impl Node {
    fn new() -> Node {
        Node{
            bias: 1.2,
            weight: 0.0,
        }
    }
}
