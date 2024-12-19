# Import necessary libraries
import pandas as pd
import numpy as np
import scipy.optimize as opt
import statsmodels.api as sm
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline  # Use this only in Jupyter Notebook

# Load the dataset
disease_df = pd.read_csv("framingham.csv")

# Drop the 'education' column
disease_df.drop(['education'], inplace=True, axis=1)

# Rename the 'male' column to 'Sex_male'
disease_df.rename(columns={'male': 'Sex_male'}, inplace=True)

# Remove NaN / NULL values
disease_df.dropna(axis=0, inplace=True)

# Display the first few rows and dataset shape
print(disease_df.head(), disease_df.shape)
print(disease_df.TenYearCHD.value_counts())

# Define features (X) and target (y)
X = np.asarray(disease_df[['age', 'Sex_male', 'cigsPerDay', 'totChol', 'sysBP', 'glucose']])
y = np.asarray(disease_df['TenYearCHD'])

# Normalize the dataset
X = preprocessing.StandardScaler().fit(X).transform(X)

# Split dataset into Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=4)

print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)

# Visualization: Countplot of CHD cases
plt.figure(figsize=(7, 5))
sns.countplot(x='TenYearCHD', data=disease_df, palette="BuGn_r")
plt.title("Distribution of TenYearCHD Cases")
plt.show()

# Train Logistic Regression Model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

# Evaluate the Model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print('Accuracy of the model is =', accuracy_score(y_test, y_pred))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
conf_matrix = pd.DataFrame(data=cm, 
                           columns=['Predicted:0', 'Predicted:1'], 
                           index=['Actual:0', 'Actual:1'])

plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Greens")
plt.title("Confusion Matrix")
plt.show()

# Print Classification Report
print('The details for confusion matrix is:')
print(classification_report(y_test, y_pred))
