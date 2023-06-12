pub fn sigmoid(input: f32) -> f32 {
    1.0f32 / (1.0f32 + f32::exp(-input))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_test() {
        assert_eq!(sigmoid(0f32), 0.5f32);
        assert!(sigmoid(-6.0) < 0.01, "negative x isn't converging to 0");
    }
}
