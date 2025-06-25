use smartcore::linalg::basic::arrays::Array;
// DenseMatrix definition
use smartcore::linalg::basic::matrix::DenseMatrix;
// KNNClassifier
use smartcore::neighbors::knn_classifier::*;
// Various distance metrics
use serde::{Deserialize, Serialize};
use smartcore::metrics::distance::*;

use csv::{Reader, StringRecord};
use smartcore::metrics::accuracy;
use std::error::Error;
use std::fs::File;
use std::io::{self, BufReader};
use std::path::Path;

#[derive(Deserialize, Serialize, Debug, Clone)]
struct Email {
    spam: String,
    real: String,
}

fn count_special_chars(s: &str) -> usize {
    s.chars().filter(|c| !c.is_alphanumeric()).count()
}
fn process_data() -> Result<(DenseMatrix<f64>, Vec<i32>), Box<dyn Error>> {
    let filename = "public/spam.csv";

    let file = File::open(filename)?;
    let mut rdr = csv::Reader::from_reader(file);

    let mut data: Vec<f64> = Vec::new();
    let mut labels: Vec<i32> = Vec::new();
    let mut rows = 0;

    for result in rdr.records() {
        let record = result?;
        let email: Email = record.deserialize(None)?;
        println!("{:?}", email);

        // features for spam email
        data.push(email.spam.len() as f64);
        data.push(count_special_chars(&email.spam) as f64);

        labels.push(1.0 as i32);
        // features for spam email
        data.push(email.real.len() as f64);
        data.push(count_special_chars(&email.real) as f64);
        labels.push(0.0 as i32);

        rows += 2;
    }

    let matrix = DenseMatrix::new(rows, 2, data, true);

    println!("{:?}", matrix);
    println!("{:?}", labels);
    Ok((matrix, labels))
}

fn process_email(email: &str) -> Result<(), Box<dyn Error>> {
    let features = vec![email.len() as f64, count_special_chars(email) as f64];
    let (matrix, labels) = process_data()?;

    let knn = KNNClassifier::fit(&matrix, &labels, Default::default())?;

    // Convert features to a slice of slices
    let feature_slice = [&features[..]];
    let prediction = knn.predict(&DenseMatrix::from_2d_array(&feature_slice))?;
    println!("Prediction: {:?}", prediction);

    if prediction[0] == 1 {
        println!("{} is predicted as SPAM", email);
    } else {
        println!("{} is predicted as REAL", email);
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let (matrix, labels) = process_data()?;

    // Split data into training and testing sets (for simplicity, using all data for training)
    let train_data = matrix.clone();
    let train_labels = labels.clone();

    // Train KNN classifier
    let knn = KNNClassifier::fit(&train_data, &train_labels, Default::default())?;

    // Predict on the training data
    let predictions = knn.predict(&train_data)?;

    // Calculate accuracy
    let acc = accuracy(&train_labels, &predictions);
    println!("Accuracy: {}", acc);
    let test_email = "winner@gmail.com";

    let result  = process_email(test_email);
   match result {
        Ok(_) => println!("Email processed successfully."),
        Err(e) => eprintln!("Error processing email: {}", e),
    }

    // Example of processing a single
    

    Ok(())
}
