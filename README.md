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

