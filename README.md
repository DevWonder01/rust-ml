# Rust Email Spam Classifier

This project is a simple email spam classifier written in Rust using the [smartcore](https://github.com/smartcorelib/smartcore) machine learning library. It demonstrates basic feature engineering, model training, and evaluation for classifying email addresses as spam or real.

## Features

- Reads a CSV file containing pairs of spam and real email addresses.
- Extracts features from each email address:
  - Length of the email address.
  - Number of special (non-alphanumeric) characters.
- Trains a K-Nearest Neighbors (KNN) classifier to distinguish between spam and real emails.
- Prints the accuracy of the model on the training data.

## Dataset

The dataset should be a CSV file located at `public/spam.csv` with the following format:

```csv
spam,real
winner@spammy.com,john.doe@example.com
cheapmeds@pharmacy.com,info@company.com
...
```

Each row contains a spam email and a real email.

## Usage

1. **Clone the repository and navigate to the project directory.**

2. **Add your dataset**  
   Place your `spam.csv` file in the `public/` directory.

3. **Build and run the project:**

   ```sh
   cargo run
   ```

4. **Output**  
   The program will print:
   - The extracted feature matrix and labels.
   - The accuracy of the KNN classifier on the training data.

## Example Output

```
Email { spam: "winner@spammy.com", real: "john.doe@example.com" }
...
DenseMatrix { ... }
[1, 0, 1, 0, ...]
Accuracy: 1
```

## Notes

- The model uses only simple features (length and special character count). For better results, consider more advanced feature engineering.
- The current setup trains and tests on the same data, which can lead to overfitting and misleadingly high accuracy. For real applications, split your data into training and testing sets.

## Dependencies

- [smartcore](https://crates.io/crates/smartcore)
- [serde](https://crates.io/crates/serde)
- [csv](https://crates.io/crates/csv)

Add them to your `Cargo.toml`:

```toml
smartcore = "0.3"
serde = { version = "1.0", features = ["derive"] }
csv = "1.1"
```

##