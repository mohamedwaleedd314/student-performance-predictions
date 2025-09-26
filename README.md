
# 🎓 Student Performance Prediction (Multiple Linear Regression)

This project demonstrates the use of **Multiple Linear Regression** to predict student performance (Performance Index) based on various features such as study hours, previous scores, sleep hours, extracurricular activities, and practice papers solved.  

---

## 📂 Dataset
- **Source:** [Kaggle - Student Performance (Multiple Linear Regression)](https://www.kaggle.com/)  
- **Rows:** 10,000  
- **Columns:**
  - `Hours Studied`
  - `Previous Scores`
  - `Extracurricular Activities`
  - `Sleep Hours`
  - `Sample Question Papers Practiced`
  - `Performance Index` (Target)

---

## 🛠️ Steps Performed

### 1. **Data Exploration**
- Checked dataset structure with `.info()` and `.describe()`.
- Observed numerical ranges and categorical values.

### 2. **Outlier Detection & Removal**
- Used **IQR method** on `Performance Index`.
- Removed data points outside the calculated bounds.

```python
lowerbound=df['Performance Index'].quantile(0.25)
upperbound=df['Performance Index'].quantile(0.75)
iqr=upperbound-lowerbound
lowerbound=lowerbound-(1.5*iqr)
upperbound=upperbound+(1.5*iqr)
df=df[(df['Performance Index']>=lowerbound) & (df['Performance Index']<=upperbound)]
```

- Visualized outliers with **boxplot**:
```python
sns.boxplot(df['Performance Index'])
```

---

### 3. **Data Preprocessing**
- Encoded categorical feature (`Extracurricular Activities`) using `LabelEncoder`.

```python
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['Extracurricular Activities'] = encoder.fit_transform(df['Extracurricular Activities'])
```

- Split into train and test sets (`80% train, 20% test`).

---

### 4. **Model Training**
- Used **Linear Regression** from `sklearn`.
- Trained the model on the training data.

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
```

---

### 5. **Model Evaluation**
- **Metrics:**
  - Mean Squared Error (MSE): `4.08`
  - R² Score: `0.989`

```python
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

- **Train/Test Scores:**
  - Train: **98.87%**
  - Test: **98.90%**

---

### 6. **Visualization**
Scatter plot of **Actual vs Predicted values**:

```python
plt.scatter(y_test, y_pred, color="blue", alpha=0.6, label="Predicted vs Actual")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color="red", linewidth=2, label="Perfect Prediction Line")
```

📊 **Result:** Predictions align very closely with actual values.

---

## 📈 Results
- The model achieved **very high accuracy (≈99%)**.
- Strong correlation between actual and predicted performance scores.
- Demonstrates effectiveness of **Multiple Linear Regression** for educational data analysis.

---

## 🚀 Tech Stack
- **Python**
- **Pandas, NumPy**
- **Matplotlib, Seaborn**
- **Scikit-learn**

---

## 📌 Future Work
- Apply **Regularization (Ridge/Lasso)** to avoid overfitting.
- Try **Polynomial Regression** for non-linear relationships.
- Deploy as a **web app (Flask/Streamlit)** for student input and prediction.

---

## 👨‍💻 Author
- **Mohamed Waleed**  
AI & Software Engineer | Data Science & Machine Learning Enthusiast  
