use crate::neural_network::matrix::Matrix;

pub type NetworkValueType = f32;

type WeightMatrix = Matrix<NetworkValueType>;

pub struct NeuralNetworkSpec {
    pub input_size: usize,
    pub output_size: usize,
    pub hidden_layers: Vec<usize>,
}

#[derive(Debug)]
pub struct NeuralNetworkWeights {
    weights: Vec<WeightMatrix>,
}

pub fn create_initial_weights(spec: &NeuralNetworkSpec) -> NeuralNetworkWeights {
    let num_layers = spec.hidden_layers.len() + 1;
    let mut weights = NeuralNetworkWeights {
        weights: Vec::with_capacity(num_layers),
    };

    for i in 0..num_layers {
        if i == 0 {
            weights.weights.push(Matrix::init_random(
                spec.input_size,
                if num_layers == 1 {
                    spec.output_size
                } else {
                    spec.hidden_layers[0]
                },
                -1.0..1.0,
            ));
        } else if i == num_layers - 1 {
            weights.weights.push(Matrix::init_random(
                spec.hidden_layers[i - 1],
                spec.output_size,
                -1.0..1.0,
            ));
        } else {
            weights.weights.push(Matrix::init_random(
                spec.hidden_layers[i - 1],
                spec.hidden_layers[i],
                -1.0..1.0,
            ));
        }
    }

    weights
}

pub fn feed_forward(
    input: &Vec<NetworkValueType>,
    weights: &NeuralNetworkWeights,
) -> Vec<NetworkValueType> {
    let mut current_vals = Matrix::from(1, input.len(), input.clone());

    for weight in weights.weights.clone() {
        current_vals = weight * current_vals;
    }

    current_vals.get_vals()
}

pub fn predict(input: &Vec<NetworkValueType>, weights: &NeuralNetworkWeights) -> usize {
    let output = feed_forward(input, weights);
    let mut best_index: usize = 0;
    let mut highest_value: NetworkValueType = 0.0;
    for (i, val) in output.into_iter().enumerate() {
        if val > highest_value {
            best_index = i;
            highest_value = val;
        }
    }
    best_index
}
