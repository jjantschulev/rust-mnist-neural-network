pub mod neural_network;
extern crate num_cpus;

use neural_network::mnist_io;
use neural_network::neural_network::*;

fn main() {
    // _xor();
    _mnist_images();
}

fn _xor() {
    let training_data_set = vec![
        GenericTrainingItem {
            inputs: vec![1.0, 1.0],
            outputs: vec![0.0],
        },
        GenericTrainingItem {
            inputs: vec![0.0, 1.0],
            outputs: vec![1.0],
        },
        GenericTrainingItem {
            inputs: vec![1.0, 0.0],
            outputs: vec![1.0],
        },
        GenericTrainingItem {
            inputs: vec![0.0, 0.0],
            outputs: vec![0.0],
        },
    ];

    let network_spec: NetworkSpec = NetworkSpec {
        hidden_layers: vec![2, 2],
        input_size: 2,
        output_size: 1,
    };

    let weights = match read_weights_from_file("xor-weights.json") {
        Ok(w) => w,
        Err(_) => {
            let mut w = create_initial_weights(&network_spec);
            train(&mut w, &training_data_set, 10000, 0.5, 256);
            w
        }
    };

    println!("\nTesting");
    for item in &training_data_set {
        let (output, _) = feed_forward(&item.inputs, &weights);
        let cost = calculate_cost(&output, &item.outputs);
        println!(
            "input: {:?}, actual: {:?}, prediction: {:?}, cost: {}",
            item.inputs, item.outputs, output, cost,
        );
    }

    save_weights_to_file(&weights, "xor-weights.json");
}

fn _mnist_images() {
    let training_image_set = mnist_io::load_image_set(
        "data/train-images-idx3-ubyte",
        "data/train-labels-idx1-ubyte",
    )
    .expect("Error loading training data");
    println!("Loaded {} training images", training_image_set.len());

    let testing_image_set =
        mnist_io::load_image_set("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte")
            .expect("Error loading training data");
    println!("Loaded {} testing images", testing_image_set.len());

    let network_spec: NetworkSpec = NetworkSpec {
        hidden_layers: vec![800, 128, 64],
        input_size: 28 * 28,
        output_size: 10,
    };

    let mut weights = match read_weights_from_file("mnist-weights.json") {
        Ok(w) => w,
        Err(_) => create_initial_weights(&network_spec),
    };
    println!("\nTraining");

    for _ in 0..0 {
        let start = std::time::Instant::now();
        train(&mut weights, &training_image_set, 10, 0.01, 1024 * 2);
        let end = std::time::Instant::now();
        println!("Training took: {:?}", end - start);

        save_weights_to_file(&weights, "mnist-weights.json");
        println!("Saved Weights");
    }

    println!("\nTesting Test Images");
    print_testing_result(&weights, &testing_image_set);
    println!("\nTesting Training Images");
    print_testing_result(&weights, &training_image_set);
}

fn print_testing_result<T: TrainingItem + Send + Sync>(
    weights: &NetworkWeights,
    testing_set: &[T],
) {
    let num_correct = test(weights, &testing_set);

    println!(
        "Num Correct: {}, Total: {}, Percentage: {}%",
        num_correct,
        testing_set.len(),
        num_correct as f32 / testing_set.len() as f32 * 100.0,
    );
}
