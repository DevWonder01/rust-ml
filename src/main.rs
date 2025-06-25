use smartcore::linalg::basic::arrays::Array;
// DenseMatrix definition
use smartcore::linalg::basic::matrix::DenseMatrix;
// KNNClassifier
use smartcore::neighbors::knn_classifier::*;
// Various distance metrics
use smartcore::metrics::distance::*;
use serde::{Deserialize, Serialize};

use std::error::Error;
use std::fs::File;
use std::io::{self, BufReader, Read}; // Import Read trait
use std::path::Path;
use csv::{Reader, StringRecord};
use smartcore::metrics::accuracy;

#[derive(Deserialize,Serialize,Debug,Clone)]
struct Email {
    // Assuming the CSV has headers "spam" and "real"
    // Adjust these field names if your CSV headers are different
    spam : String,
    real : String // This field seems to represent a 'real' email example in the dataset row
}

// Function to count non-alphanumeric characters
fn count_special_chars(s: &str) -> usize {
    s.chars().filter(|c| !c.is_alphanumeric()).count()
}

// Function to process the CSV data and extract features and labels
fn process_data() ->Result<(DenseMatrix<f64>, Vec<i32>), Box<dyn Error>>{

    // Assuming the CSV file is named "spam.csv" and is in a "public" directory
    let filename = "public/spam.csv";

    let file = File::open(filename)?;
    // Use from_reader directly with the file
    let mut rdr = csv::Reader::from_reader(file);

    let mut data: Vec<f64> = Vec::new();
    let mut labels: Vec<i32> = Vec::new();
    let mut rows = 0;

    // Iterate over CSV records
    for result in rdr.records(){
        let record = result?;
        // Deserialize the record into the Email struct
        let email: Email = record.deserialize(None)?;

        // Extract features for the 'spam' example in the row
        // Feature 1: Length of the 'spam' string
        data.push(email.spam.len() as f64);
        // Feature 2: Count of special characters in the 'spam' string
        data.push(count_special_chars(&email.spam) as f64);
        // Label for 'spam' is 1
        labels.push(1);

        // Extract features for the 'real' example in the row
        // Feature 1: Length of the 'real' string
        data.push(email.real.len() as f64);
        // Feature 2: Count of special characters in the 'real' string
        data.push(count_special_chars(&email.real) as f64);
        // Label for 'real' is 0
        labels.push(0);

        // Each record in the CSV contributes two data points (one spam, one real)
        rows +=2;
    }

    // Create a DenseMatrix from the collected data
    // `rows` is the number of samples, 2 is the number of features
    let matrix = DenseMatrix::new(rows, 2, data,true);

    println!("Processed Data Matrix: {:?}",matrix);
    println!("Processed Labels: {:?}",labels);
    Ok((matrix, labels))
}

// Function to check if an email address is fake using the trained KNN model
fn is_fake_email(knn: &KNNClassifier<f64, i32, DenseMatrix<f64>, Distances<f64>>, email_address: &str) -> Result<i32, Box<dyn Error>> {
    // Calculate the same features used for training
    let email_len = email_address.len() as f64;
    let special_chars_count = count_special_chars(email_address) as f64;

    // Create a DenseMatrix for the single email address
    // The matrix should have 1 row and 2 columns (for the two features)
    let input_features = DenseMatrix::from_array(&[
        &[email_len, special_chars_count]
    ]);

    // Use the trained KNN model to predict
    let predictions = knn.predict(&input_features)?;

    // The predict method returns a Vec<i32> with one prediction in this case
    // Return the first (and only) prediction
    predictions.into_iter().next().context("Failed to get prediction")
}


fn main() -> Result<(), Box<dyn Error>> {
    // Process the data to get the training matrix and labels
    let (matrix, labels) = process_data()?;

    // Split data into training and testing sets (for simplicity, using all data for training)
    // In a real-world scenario, you would split your data into distinct training and testing sets
    let train_data = matrix.clone();
    let train_labels = labels.clone();

    // Train KNN classifier
    // Default::default() uses default parameters for KNN (e.g., k=5, Euclidian distance)
    let knn = KNNClassifier::fit(&train_data, &train_labels, Default::default())?;

    // Predict on the training data to evaluate the model's performance on seen data
    let training_predictions = knn.predict(&train_data)?;

    // Calculate accuracy on the training data
    let acc = accuracy(&train_labels, &training_predictions);
    println!("Training Accuracy: {}", acc);

    // --- Example Usage of the new `is_fake_email` function ---

    let test_email_real = "test@example.com";
    let prediction_real = is_fake_email(&knn, test_email_real)?;
    println!("Email: '{}' -> Predicted Label: {}", test_email_real, prediction_real);
    // Expected output for a real email (based on features): 0

    let test_email_spam = "!!!!win-a-prize!!!! click here now!!!";
    let prediction_spam = is_fake_email(&knn, test_email_spam)?;
    println!("Email: '{}' -> Predicted Label: {}", test_email_spam, prediction_spam);
    // Expected output for a spam email (based on features): 1

    let test_email_mixed = "hello! check out this link: example.com";
    let prediction_mixed = is_fake_email(&knn, test_email_mixed)?;
    println!("Email: '{}' -> Predicted Label: {}", test_email_mixed, prediction_mixed);
    // The prediction for this will depend on how similar its features (length, special chars)
    // are to the trained spam/real examples in your CSV and the KNN algorithm's decision boundary.


    Ok(())
}
