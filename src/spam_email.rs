// filepath: /C:/Users/jimmy/OneDrive/Documents/Workstation/Projects/AI/rust-ml/src/main.rs
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::neighbors::knn_classifier::{KNNClassifier, KNNClassifierParameters};
use smartcore::metrics::accuracy;
use smartcore::math::distance::euclidian::Euclidian;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs::File;
use std::io::{self, Write, BufReader, BufWriter};
use csv::Reader;
use bincode;

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
    let mut rdr = Reader::from_reader(file);

    let mut data: Vec<f64> = Vec::new();
    let mut labels: Vec<i32> = Vec::new();
    let mut rows = 0;

    for result in rdr.records() {
        let record = result?;
        let email: Email = record.deserialize(None)?;

        // Features for spam email
        data.push(email.spam.len() as f64);
        data.push(count_special_chars(&email.spam) as f64);
        labels.push(1); // Label for spam email

        // Features for real email
        data.push(email.real.len() as f64);
        data.push(count_special_chars(&email.real) as f64);
        labels.push(0); // Label for real email

        rows += 2;
    }

    let matrix = DenseMatrix::new(rows, 2, data);
    Ok((matrix, labels))
}

fn predict_email(knn: &KNNClassifier<f64, DenseMatrix<f64>, Vec<i32>, Euclidian>, email: &str) -> Result<i32, Box<dyn Error>> {
    let features = vec![
        email.len() as f64,
        count_special_chars(email) as f64,
    ];
    let matrix = DenseMatrix::new(1, 2, features);
    let prediction = knn.predict(&matrix)?;
    Ok(prediction[0])
}

fn save_model<T: Serialize>(model: &T, path: &str) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    bincode::serialize_into(writer, model)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let (matrix, labels) = process_data()?;

    // Split data into training and testing sets (for simplicity, using all data for training)
    let train_data = matrix.clone();
    let train_labels = labels.clone();

    // Train KNN classifier
    let knn = KNNClassifier::fit(&train_data, &train_labels, KNNClassifierParameters::default())?;

    // Predict on the training data
    let predictions = knn.predict(&train_data)?;

    // Calculate accuracy
    let acc = accuracy(&train_labels, &predictions);
    println!("Accuracy: {}", acc);

    // Save the trained model to a file
    save_model(&knn, "knn_model.bin")?;
    println!("Model saved to knn_model.bin");

    // Test with random emails
    let test_emails = vec![
        "random@unknown.com",
        "winner@spammy.com",
        "contact@website.com",
    ];

    for email in test_emails {
        let prediction = predict_email(&knn, email)?;
        let label = if prediction == 1 { "Spam" } else { "Real" };
        println!("Email: {}, Prediction: {}", email, label);
    }

    Ok(())
}