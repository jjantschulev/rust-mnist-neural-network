use crate::neural_network::matrix::Matrix;
use crate::neural_network::parallel_worker;
use rand::thread_rng;
use rand::Rng;
use serde::{Deserialize, Serialize};

pub type NetworkValue = f32;

type NetworkMatrix = Matrix<NetworkValue>;
type NetworkLayerState = (Vec<NetworkValue>, NetworkMatrix, Vec<NetworkValue>);

pub struct NetworkSpec {
    pub input_size: usize,
    pub output_size: usize,
    pub hidden_layers: Vec<usize>,
}

pub type NetworkWeights = Vec<WeightsAndBiases>;

#[derive(Debug, Serialize, Deserialize)]
pub struct WeightsAndBiases {
    weights: NetworkMatrix,
    biases: NetworkMatrix,
}

pub trait TrainingItem {
    fn inputs(&self) -> &[NetworkValue];
    fn outputs(&self) -> Vec<NetworkValue>;
}

pub struct GenericTrainingItem {
    pub inputs: Vec<NetworkValue>,
    pub outputs: Vec<NetworkValue>,
}

impl TrainingItem for GenericTrainingItem {
    fn inputs(&self) -> &[NetworkValue] {
        &self.inputs
    }
    fn outputs(&self) -> Vec<NetworkValue> {
        self.outputs.clone()
    }
}

pub fn create_initial_weights(spec: &NetworkSpec) -> NetworkWeights {
    let num_layers = spec.hidden_layers.len() + 1;
    let mut weights = NetworkWeights::with_capacity(num_layers);

    for i in 0..num_layers {
        let biases = Matrix::init_random(
            1,
            if i == num_layers - 1 {
                spec.output_size
            } else {
                spec.hidden_layers[i]
            },
            -1.0..1.0,
        );

        if i == 0 {
            weights.push(WeightsAndBiases {
                biases,
                weights: Matrix::init_random(
                    spec.input_size,
                    if num_layers == 1 {
                        spec.output_size
                    } else {
                        spec.hidden_layers[0]
                    },
                    -1.0..1.0,
                ),
            });
        } else if i == num_layers - 1 {
            weights.push(WeightsAndBiases {
                biases,
                weights: Matrix::init_random(
                    spec.hidden_layers[i - 1],
                    spec.output_size,
                    -1.0..1.0,
                ),
            });
        } else {
            weights.push(WeightsAndBiases {
                biases,
                weights: Matrix::init_random(
                    spec.hidden_layers[i - 1],
                    spec.hidden_layers[i],
                    -1.0..1.0,
                ),
            });
        }
    }

    weights
}

pub fn feed_forward(
    input: &[NetworkValue],
    weights: &NetworkWeights,
) -> (Vec<NetworkValue>, Vec<NetworkLayerState>) {
    let mut current_vals: NetworkMatrix = Matrix::new(1, input.len());
    for (i, val) in input.iter().enumerate() {
        current_vals.set(0, i, *val);
    }
    let mut network_state: Vec<NetworkLayerState> = Vec::with_capacity(0);

    for WeightsAndBiases { weights, biases } in weights {
        network_state.push((
            current_vals.clone().into_vals(),
            weights.clone(),
            biases.clone().into_vals(),
        ));
        current_vals = Matrix::multiply_sync(weights, &current_vals);
        current_vals.map(|v, i| sigmoid(*v + *biases.get(0, i).unwrap()));
    }

    (current_vals.into_vals(), network_state)
}

pub fn predict(input: &[NetworkValue], weights: &NetworkWeights) -> (usize, Vec<NetworkValue>) {
    let (output, _) = feed_forward(input, weights);
    (get_index_of_max_value(&output).unwrap(), output)
}

fn get_index_of_max_value(array: &[NetworkValue]) -> Option<usize> {
    let mut best_index: Option<usize> = None;
    let mut highest_value: Option<NetworkValue> = None;
    for (i, val) in array.iter().enumerate() {
        if highest_value.is_none() || *val > *highest_value.as_ref().unwrap() {
            best_index = Some(i);
            highest_value = Some(*val);
        }
    }
    best_index
}

pub fn train<T: TrainingItem + Send + Sync>(
    weights: &mut NetworkWeights,
    training_data: &[T],
    training_steps: usize,
    learning_rate: NetworkValue,
    batch_size: usize,
) {
    // let network_cost = calculate_network_cost(weights, training_data);
    // println!("Total Network Cost Before Training: {}", network_cost);

    const NUM_PRINT_STEPS: usize = 10;

    for i in 0..training_steps {
        let mut training_item_indicies: Vec<usize> = vec![0; batch_size];
        for j in 0..batch_size {
            training_item_indicies[j] = thread_rng().gen_range(0..training_data.len());
        }
        // println!("Training Indicies: {:?}", training_item_indicies);

        let gradients = parallel_worker::map(
            parallel_worker::divide_tasks_into_chunks(&training_item_indicies[0..batch_size]),
            |batch| train_on_batch(weights, &training_data, batch),
        )
        .unwrap();
        let num_threads = gradients.len() as NetworkValue;

        for gradient in gradients {
            apply_gradient_to_network(weights, gradient, -learning_rate / num_threads);
        }

        if training_steps <= NUM_PRINT_STEPS || (i + 1) % (training_steps / NUM_PRINT_STEPS) == 0 {
            let cost_sample_size = std::cmp::min(batch_size, training_data.len());
            let cost_start_point = if training_data.len() == cost_sample_size {
                0
            } else {
                thread_rng().gen_range(0..(training_data.len() - batch_size))
            };
            let range = cost_start_point..(cost_start_point + cost_sample_size);
            let network_cost = calculate_network_cost(weights, &training_data[range]);
            println!(
                "Finished Training Batch: {} of {}. Cost: {}",
                i + 1,
                training_steps,
                network_cost
            );
        }
    }

    // let network_cost = calculate_network_cost(weights, training_data);
    // println!("Total Network Cost After Training: {}", network_cost);
}

pub fn test<T: TrainingItem + Send + Sync>(weights: &NetworkWeights, training_data: &[T]) -> usize {
    parallel_worker::map_reduce(
        training_data,
        |item| {
            if predict(item.inputs(), weights).0 == get_index_of_max_value(&item.outputs()).unwrap()
            {
                1
            } else {
                0
            }
        },
        |a, b| a + b,
        0,
    )
    .unwrap()
}

fn train_on_batch<T: TrainingItem>(
    weights: &NetworkWeights,
    training_data: &[T],
    indicies: &[usize],
) -> NetworkWeights {
    // Create Gradient
    let mut network_gradient: NetworkWeights = Vec::with_capacity(weights.len());

    for WeightsAndBiases { weights, biases } in weights {
        network_gradient.push(WeightsAndBiases {
            weights: Matrix::new(weights.width(), weights.height()),
            biases: Matrix::new(biases.width(), biases.height()),
        });
    }

    for index in indicies {
        let (output, network_state) = feed_forward(&training_data[*index].inputs(), weights);

        backpropagation(
            &training_data[*index].outputs(),
            &output,
            network_state,
            &mut network_gradient,
        );
    }

    // Calculate Average and return;
    for WeightsAndBiases { weights, biases } in network_gradient.iter_mut() {
        weights.map(|v, _| v / (indicies.len() as NetworkValue));
        biases.map(|v, _| v / (indicies.len() as NetworkValue));
    }

    return network_gradient;
}

fn backpropagation(
    target_values: &[NetworkValue],
    current_values: &[NetworkValue],
    network_state: Vec<NetworkLayerState>,
    network_gradient: &mut NetworkWeights,
) {
    let mut nudges = vec![0.0; target_values.len()];
    for i in 0..target_values.len() {
        nudges[i] = current_values[i] - target_values[i];
    }

    let mut current_values = current_values;

    for i in (0..network_state.len()).rev() {
        nudges = backpropagation_step(
            &nudges,
            current_values,
            &network_state[i],
            &mut network_gradient[i],
        );
        current_values = &network_state[i].0;
    }
}

fn backpropagation_step(
    nudges: &[NetworkValue],
    current_values: &[NetworkValue],
    (parent_vals, parent_weights, parent_biases): &NetworkLayerState,
    WeightsAndBiases {
        weights: parent_gradient_weights,
        biases: parent_gradient_biases,
    }: &mut WeightsAndBiases,
) -> Vec<NetworkValue> {
    // if thread_rng().gen_range(0..10000) == 1 {
    //     println!("Nudges: {:?}", nudges);
    // }
    // Update Parent Layer Weights
    for y in 0..parent_weights.height() {
        for x in 0..parent_weights.width() {
            let dw = parent_vals[x]
                * ((current_values[y]) * (1.0 - current_values[y]))
                * 2.0
                * nudges[y];
            parent_gradient_weights.set(x, y, parent_gradient_weights.get(x, y).unwrap() + dw);
        }
    }

    // Update Parent Layer Biases
    for i in 0..parent_biases.len() {
        let db = ((current_values[i]) * (1.0 - current_values[i])) * 2.0 * nudges[i];
        parent_gradient_biases.set(0, i, parent_gradient_biases.get(0, i).unwrap() + db);
    }

    // Return Parent Layer Value Nudges
    let mut parent_layer_nudges: Vec<NetworkValue> = vec![0.0; parent_vals.len()];

    for i in 0..parent_vals.len() {
        let mut dp = 0.0;

        for j in 0..current_values.len() {
            dp += parent_weights.get(i, j).unwrap()
                // * ((current_values[j]) * (1.0 - current_values[j]))
                * 2.0
                * nudges[j]
        }

        parent_layer_nudges[i] = dp;
    }

    parent_layer_nudges
}

pub fn calculate_network_cost<T: TrainingItem + Send + Sync>(
    weights: &NetworkWeights,
    data: &[T],
) -> NetworkValue {
    let mut average_cost: NetworkValue = 0.0;

    let values = parallel_worker::map_reference(data, |training_item| {
        let (prediction, _) = feed_forward(&training_item.inputs(), &weights);
        calculate_cost(&prediction, &training_item.outputs())
    })
    .unwrap();

    for value in values {
        average_cost += value;
    }

    average_cost / (data.len() as NetworkValue)
}

pub fn calculate_cost(
    network_output: &[NetworkValue],
    expected_outputs: &[NetworkValue],
) -> NetworkValue {
    let mut error: NetworkValue = 0.0;

    for (actual, expected) in network_output.iter().zip(expected_outputs) {
        let single_error = actual - expected;
        error += (single_error) * (single_error);
    }

    error
}

fn sigmoid(x: NetworkValue) -> NetworkValue {
    return 1.0 / (1.0 + std::f32::consts::E.powf(-x));
}

fn apply_gradient_to_network(
    weights: &mut NetworkWeights,
    gradient: NetworkWeights,
    learning_rate: NetworkValue,
) {
    for (layer_index, WeightsAndBiases { weights, biases }) in weights.iter_mut().enumerate() {
        weights.map(|v, i| v + gradient[layer_index].weights.vals()[i] * learning_rate);
        biases.map(|v, i| v + gradient[layer_index].biases.vals()[i] * learning_rate);
    }
}

pub fn save_weights_to_file(weights: &NetworkWeights, file_name: &str) {
    let string = serde_json::to_string(weights).unwrap();
    std::fs::write(file_name, string).unwrap();
}

pub fn read_weights_from_file(file_name: &str) -> Result<NetworkWeights, ()> {
    let file = match std::fs::read_to_string(file_name) {
        Ok(d) => d,
        Err(_) => return Err(()),
    };
    match serde_json::from_str(&file) {
        Ok(d) => Ok(d),
        Err(_) => Err(()),
    }
}
