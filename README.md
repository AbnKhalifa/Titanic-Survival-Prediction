# 🚢 Titanic Survival Prediction – Neural Network

This project uses a neural network built with TensorFlow/Keras to predict passenger survival on the Titanic dataset. The goal is to explore classification techniques, data preprocessing, and model evaluation in a real-world scenario.

---

## 📊 Problem Statement

Given information about passengers such as age, gender, class, and fare, predict whether they survived the Titanic disaster or not.

- Type: Binary Classification (`Survived`: 0 = No, 1 = Yes)
- Dataset: [Titanic - Machine Learning from Disaster (Kaggle)](https://www.kaggle.com/c/titanic)

---

## 🧠 Model Summary

A simple feedforward neural network built using Keras with:

- Input Layer: Standardized numerical and encoded categorical features
- Hidden Layers: `Dense (64) → Dropout → Dense (32)`
- Output Layer: `Dense (1)` with `sigmoid` activation
- Loss Function: Binary Crossentropy
- Optimizer: Adam
- Accuracy achieved: **~80%**

---

## 🛠️ Tech Stack

- **Language**: Python
- **Libraries**: Pandas, NumPy, Seaborn, Scikit-learn, TensorFlow/Keras, Matplotlib

---

## 📁 Project Structure

titanic_survival/
│
├── data/
│ └── train.csv
├── titanic.ipynb # Jupyter notebook with full code
├── README.md # This file
├── requirements.txt # Python packages

---

## 🔍 Key Steps

1. **Data Preprocessing**
   - Handled missing values (`Age`, `Embarked`)
   - Removed irrelevant columns (`Name`, `Cabin`, `Ticket`)
   - Converted categorical variables with one-hot encoding

2. **Feature Scaling**
   - StandardScaler used to normalize inputs for neural network

3. **Model Building & Training**
   - Neural Network with Keras Sequential API
   - Dropout used to prevent overfitting
   - Trained for 50 epochs with validation split

4. **Evaluation**
   - Accuracy Score
   - Confusion Matrix
   - Classification Report

---

## 📈 Results

- **Test Accuracy**: ~80%
- Model shows good generalization and balance between precision & recall.

---

## 📌 Future Improvements

- Hyperparameter tuning (batch size, epochs, layers)
- Try other models (Logistic Regression, Random Forest, XGBoost)
- Add cross-validation and AUC-ROC analysis
- Add test set predictions and submit to Kaggle

---

## 🙌 Author

**Adham Khalifa**  
📫 [LinkedIn](https://www.linkedin.com/in/abn-khalifa)  
💻 [GitHub](https://github.com/AbnKhalifa)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
