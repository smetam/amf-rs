use amf::common::{ClassifierTarget, Observation};
use amf::forest::AMFClassifier;
use amf::metrics::Accuracy;
use csv::Reader;
use std::fs::File;

fn main() {
    let mut accuracy = Accuracy::new();
    let mut forest = AMFClassifier::new(10, 1.0, true, 0.5, false, Some(123));

    let file_path = "examples/banana.csv";

    let Ok(file) = File::open(file_path) else {
        panic!("Failed to open")
    };

    // Create a CSV reader
    let mut reader = Reader::from_reader(file);

    // Get the header of the CSV file
    let header_record = reader.headers().unwrap();
    // Convert the header record to a vector of strings
    let column_names: Vec<&str> = header_record.iter().collect();

    let col1 = column_names[0].to_string();
    let col2 = column_names[1].to_string();
    let target = column_names[2].to_string();
    println!("{:?} {:?} {:?}", col1, col2, target);

    let default_label = ClassifierTarget::String("1".to_string());
    // Iterate over the rows of the CSV file
    for (i, result) in reader.records().enumerate() {
        // Handle errors reading a row
        let record = result.unwrap();

        let row: Vec<&str> = record.iter().collect();
        // Process the fields of the row
        let y = ClassifierTarget::String(row[2].to_string());
        let mut x = Observation::new();
        x.insert(col1.clone(), row[0].parse::<f64>().unwrap());
        x.insert(col2.clone(), row[1].parse::<f64>().unwrap());
        // println!("\n>>>>>>line: {:?}, x: {:?}, y: {:?}", i, x, y);

        let prediction = forest.predict_proba_one(&x);
        let mut predicted_class: Option<ClassifierTarget> = None;
        let mut max_value = 0.; // Initialize with the minimum value of i32

        // Iterate over key-value pairs to find the maximum value and its corresponding key
        for (key, &value) in &prediction {
            if value > max_value {
                predicted_class = Some(key.clone());
                max_value = value;
            }
        }
        let predicted_class = predicted_class.as_ref().unwrap_or(&default_label);
        accuracy.update(&y, predicted_class);
        forest.learn_one(&x, &y);
    }
    // println!("tree: {:?}", forest);
    // forest.trees[0].print_tree();

    println!("ACCURACY: {:?}", accuracy.get())
}
