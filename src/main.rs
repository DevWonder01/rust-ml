extern crate csv;
extern crate ndarray;
extern crate serde;
extern crate smartcore;

use csv::ReaderBuilder;
use ndarray::prelude::*;
use serde::Deserialize;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::naive_bayes::gaussian::GaussianNB;
use smartcore::metrics::accuracy;
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
struct Email {
    label: String,
    text: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load the dataset
    let mut rdr = ReaderBuilder::new().from_path("emails.csv")?;
    let emails: Vec<Email> = rdr.deserialize().collect::<Result<_, _>>()?;

    // Preprocess the data
    let (features, labels) = preprocess_data(&emails);

    // Convert to DenseMatrix
    let features_matrix = DenseMatrix::from_array(features.shape()[0], features.shape()[1], features.as_slice().unwrap());

    // Train the model
    let model = GaussianNB::fit(&features_matrix, &labels, Default::default()).unwrap();

    // Evaluate the model
    let predictions = model.predict(&features_matrix).unwrap();
    let accuracy = accuracy(&labels, &predictions);
    println!("Model accuracy: {:.2}%", accuracy * 100.0);

    Ok(())
}

fn preprocess_data(emails: &[Email]) -> (Array2<f64>, Vec<f64>) {
    let mut word_counts: HashMap<String, usize> = HashMap::new();
    let mut labels = Vec::new();
    let mut data = Vec::new();

    // Build the vocabulary
    for email in emails {
        for word in email.text.split_whitespace() {
            let count = word_counts.entry(word.to_string()).or_insert(0);
            *count += 1;
        }
    }

    // Convert emails to feature vectors
    for email in emails {
        let mut features = vec![0.0; word_counts.len()];
        for word in email.text.split_whitespace() {
            if let Some(&index) = word_counts.get(word) {
                features[index] += 1.0;
            }
        }
        data.push(features);
        labels.push(if email.label == "spam" { 1.0 } else { 0.0 });
    }

    let features = Array2::from_shape_vec((emails.len(), word_counts.len()), data.concat()).unwrap();
    (features, labels)
}