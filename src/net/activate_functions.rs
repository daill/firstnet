pub fn sigmoid(input: f32) -> f32 {
    1.0f32 / (1.0f32 + f32::exp(-input))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_test() {
        let result = 0.5f32;
        assert_eq!(sigmoid(0f32), result);
        let result = -1f32;
        assert!(sigmoid(-6.0) < 0.001, "negative x isn't converging to 0");
    }
}
