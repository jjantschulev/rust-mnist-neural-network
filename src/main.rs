pub mod neural_network;
extern crate num_cpus;

use neural_network::mnist_io;
use neural_network::neural_network::*;

fn main() {
    println!("Hello, world!. You have {} cpus", num_cpus::get());

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

    let network_spec: NeuralNetworkSpec = NeuralNetworkSpec {
        hidden_layers: vec![100, 100],
        input_size: 28 * 28,
        output_size: 10,
    };

    let weights = create_initial_weights(&network_spec);

    println!("Weights Generated");

    for image in &testing_image_set[1..10] {
        let result = predict(&image.pixels, &weights);
        println!("actual: {}, prediction: {:?}", image.label, result);
    }
}
