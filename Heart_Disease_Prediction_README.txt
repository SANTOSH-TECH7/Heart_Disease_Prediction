
# Heart Disease Prediction

This project predicts the likelihood of a person developing coronary heart disease (CHD) within 10 years using logistic regression. The dataset used is the Framingham Heart Study dataset.

## Dataset

The dataset `framingham.csv` contains medical and lifestyle information about individuals, including features such as:
- Age
- Gender
- Cigarettes per day
- Total cholesterol
- Systolic blood pressure
- Glucose levels

The target variable is `TenYearCHD`, indicating the presence (1) or absence (0) of CHD within 10 years.

## Steps to Perform the Process

1. **Import Necessary Libraries**:
   - Libraries such as pandas, numpy, matplotlib, seaborn, and scikit-learn are used.

2. **Load the Dataset**:
   - The dataset is loaded into a Pandas DataFrame using `pd.read_csv`.

3. **Data Preprocessing**:
   - Drop unnecessary columns such as `education`.
   - Rename the `male` column to `Sex_male` for clarity.
   - Remove rows with missing values using `dropna`.

4. **Feature Selection**:
   - The selected features for prediction are:
     - `age`, `Sex_male`, `cigsPerDay`, `totChol`, `sysBP`, `glucose`.

5. **Data Normalization**:
   - Features are normalized using `StandardScaler` to ensure all features contribute equally to the model.

6. **Train-Test Split**:
   - The dataset is split into training (70%) and testing (30%) subsets using `train_test_split`.

7. **Logistic Regression Model**:
   - A logistic regression model is trained using the training data.

8. **Model Evaluation**:
   - Evaluate the model using:
     - Accuracy score
     - Confusion matrix
     - Classification report

9. **Visualization**:
   - Plot the distribution of CHD cases using a countplot.
   - Visualize the confusion matrix using a heatmap.

## Code Example

Refer to the script in this repository for the full implementation.

## Results

- **Accuracy**: Displays the percentage of correct predictions.
- **Confusion Matrix**: Shows true positives, true negatives, false positives, and false negatives.
- **Classification Report**: Provides precision, recall, F1-score, and support for each class.

## Requirements

Install the following Python libraries before running the script:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Run the Script

1. Clone this repository or download the script.
2. Place the `framingham.csv` file in the same directory as the script.
3. Run the script to train the model and visualize the results.
