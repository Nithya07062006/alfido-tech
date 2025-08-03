# alfido-tech
TASK 1::::
# 🧹 AI Data Preprocessing Pipeline – Python

This project demonstrates a complete data preprocessing workflow using **Pandas** and **Scikit-learn**, preparing raw data for AI/ML model training without requiring any external dataset.

---

## 📌 Features

- 🗃️ Creates a sample dataset (no CSV upload needed)
- 🧹 Handles missing values via mean imputation
- 🔠 Encodes categorical features using `LabelEncoder`
- 📏 Normalizes features using `StandardScaler`
- ✂️ Splits data into training and testing sets using `train_test_split`

---

## 🎯 Objective

Preprocessing is a critical step in any machine learning pipeline. Clean and well-structured data leads to better-performing AI models. This project shows how to:
- Clean missing values  
- Encode categorical variables  
- Scale numeric values  
- Prepare data for training and evaluation  

---

## 🛠️ Tech Stack

- **Python 3**
- **Pandas**
- **Scikit-learn**

---

## 🚀 How to Run

1. Clone this repository  
2. Open `preprocessing.py` or copy the code into a Jupyter Notebook / Google Colab  
3. Run the script  
4. ✅ No data files needed – it uses a synthetic dataset

---

## 📌 Sample Output

Initial Data:
Name Age Salary Gender Purchased
0 Alice 25.0 50000.0 Female Yes
...

Encoded Data:
Name Age Salary Gender Purchased
0 0 25.0 50000.0 0 1
...

Normalized Data:
Name Age Salary Gender Purchased
0 -1.414214 -1.507557 -1.482617 -0.912871 1.224745



---

## 🤖 Future Work

- Integrate with real-world datasets  
- Add outlier detection and removal  
- Use this as preprocessing for ML model training (e.g., Logistic Regression, SVM)



## 🙌 Author

**Nithyashree H S**  
Passionate about AI, Web Dev & Full Stack Development  
📫 [LinkedIn](www.linkedin.com/in/nithya-shree-8802542b3) 

---

## 📎 License

This project is licensed under the MIT License - feel free to use and modify it!

 Task 2 ==
 – Data Preprocessing & Classification Models (Alfido Internship)
This project demonstrates how to handle real-world data preprocessing and implement two classic classification models: Logistic Regression and Decision Tree Classifier.

📌 Project Description
The dataset is a small sample of customer records with attributes like Name, Age, Salary, Gender, and Purchase Decision. The task involves:

Handling missing data

Encoding categorical variables

Scaling features

Splitting the dataset

Training classification models

Evaluating performance with accuracy, confusion matrix, and classification report

🔧 Technologies Used
Python 🐍

Pandas

Scikit-learn (sklearn)

Jupyter Notebook or any Python IDE

⚙️ Steps Performed
Data Creation
A custom dataset with missing values and categorical features.

Data Preprocessing

Filled missing values using mean imputation

Encoded categorical columns with LabelEncoder

Scaled numerical features using StandardScaler

Train-Test Split

80% training

20% testing (random_state=42 for reproducibility)

Model Training

Logistic Regression

Decision Tree Classifier

Model Evaluation

Accuracy

Confusion Matrix

Classification Report

📊 Results Summary
Both models were evaluated and printed metrics such as:

Model Accuracy

Confusion Matrix

Precision, Recall, F1-Score

✅ Learning Outcomes
Data preprocessing is a critical step before model training

Logistic Regression is great for baseline binary classification

Decision Trees handle non-linear data and categorical splits well

Model evaluation metrics help understand real-world model behavior

📄 Author
Nithyashree HS
Intern – Alfido Internship Program
Task 2 Completion

 Task 3=
 – Sentiment Analysis using NLTK and TextBlob
This project demonstrates how to perform sentiment analysis on text data using two popular Python libraries: NLTK (VADER) and TextBlob. It was completed as part of the Alfido internship program.

📌 Project Overview
The task involved:

Using VADER from the nltk.sentiment module to analyze sentiment (positive, negative, neutral)

Using TextBlob to compute polarity and subjectivity of given text samples

This dual approach provides two perspectives on sentiment classification — rule-based (VADER) and probabilistic/statistical (TextBlob).

🔧 Tools & Libraries
Python 🐍

NLTK (Natural Language Toolkit)

TextBlob

Install the libraries with:

bash
Copy
Edit
pip install nltk textblob
⚙️ Implementation Steps
Import libraries
nltk, SentimentIntensityAnalyzer, TextBlob

Download required lexicons

python
Copy
Edit
nltk.download('vader_lexicon')
Use VADER for rule-based sentiment analysis

Score: compound, pos, neu, neg

Use TextBlob for polarity & subjectivity

Polarity range: -1 (negative) to 1 (positive)

Subjectivity: 0 (objective) to 1 (subjective)

📝 Example Outputs
VADER Output:

json
Copy
Edit
{'neg': 0.0, 'neu': 0.28, 'pos': 0.72, 'compound': 0.8519}
TextBlob Output:

yaml
Copy
Edit
Polarity: -0.6, Subjectivity: 0.9
🎯 Learning Outcomes
Learned how to implement and compare two sentiment analysis methods

Understood the difference between rule-based and statistical NLP models

Practiced preprocessing, scoring, and interpreting sentiment outputs

📁 File Structure
sentiment_analysis.py – main code file (you can rename this based on your file)

Sample texts included in code

🙋 Author
Nithyashree HS
Intern – Alfido Internship
Task 3 Completion ✅


Task 4 – 

Advanced MNIST Classification Using CNN (Alfido Internship)
This project implements a Convolutional Neural Network (CNN) using TensorFlow to classify handwritten digits from the MNIST dataset, enhanced with data augmentation and model evaluation tools. This is the final and most advanced task of the internship.

📌 Project Objective
Build a robust image classification model using CNN

Improve performance through data augmentation, batch normalization, and dropout

Evaluate results using confusion matrix and classification report

🔧 Technologies & Tools
Python

TensorFlow / Keras

Matplotlib & Seaborn

Scikit-learn

NumPy

Jupyter or any Python IDE

⚙️ Key Steps
Load MNIST Data
Built-in dataset containing 28x28 grayscale images of handwritten digits (0–9)

Preprocess & Augment

Normalize pixel values

Apply rotation, zoom, and shift using ImageDataGenerator

Build CNN Model

Convolutional + MaxPooling layers

BatchNormalization and Dropout to prevent overfitting

Fully connected softmax output

Train the Model

Run for 10 epochs using augmented data

Validate against the test set

Evaluate Performance

Accuracy score

Confusion matrix (visualized with Seaborn)

Precision, recall, and F1-score

Visualize Predictions

Manually plot predictions for selected images

📊 Sample Output
yaml
Copy
Edit
✅ Final Accuracy: 0.9872

📋 Classification Report:
              precision    recall  f1-score   support
           0       0.99      0.99      0.99       980
           1       0.99      0.99      0.99      1135
           ...
           9       0.99      0.98      0.98      1000
📚 Learning Outcomes
Built a deep learning model from scratch

Understood how to preprocess and augment image data

Practiced advanced evaluation techniques

Gained confidence working with CNNs and TensorFlow

🚀 How to Run
bash
Copy
Edit
pip install tensorflow matplotlib seaborn scikit-learn
python task4_mnist_cnn.py
Make sure you have Python 3.8–3.10 for TensorFlow compatibility.

🙋 Author
Nithyashree HS
Intern – Alfido Internship Program
Completed: Task 4 ✅





