import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Title of the app
st.title("Loan Prediction Data Analysis and Model Evaluation")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the dataset
    df = pd.read_csv(uploaded_file)

    st.write("### Data Preview")
    st.dataframe(df.head())

    # Handle missing data
    st.write("### Handling Missing Data")
    df.fillna({
        'Gender': df['Gender'].mode()[0],
        'Self_Employed': df['Self_Employed'].mode()[0],
        'Married': df['Married'].mode()[0],
        'Dependents': '0',
        'LoanAmount': df['LoanAmount'].median(),
        'Loan_Amount_Term': df['Loan_Amount_Term'].median(),
        'Credit_History': df['Credit_History'].median(),
    }, inplace=True)
    df['Dependents'].replace('3+', '3', inplace=True)

    # Outlier handling
    columns_to_handle = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    for col in columns_to_handle:
        df[col] = np.clip(df[col], df[col].quantile(0.09), df[col].quantile(0.91))

    # Show updated DataFrame
    st.write("### Updated Data (After Cleaning)")
    st.dataframe(df.head())

    # Feature encoding
    label_encoded_cols = ['Gender', 'Married', 'Self_Employed']
    for col in label_encoded_cols:
        df[col] = df[col].astype('category').cat.codes

    df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
    property_area_dummies = pd.get_dummies(df['Property_Area'], prefix='Property_Area')
    df = pd.concat([df.drop('Property_Area', axis=1), property_area_dummies], axis=1)

    # Train-test split
    X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
    y = df['Loan_Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Models and evaluations
    st.write("### Model Evaluation")

    # Decision Tree
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)
    dt_acc = accuracy_score(y_test, dt_model.predict(X_test))
    st.write(f"**Decision Tree Accuracy:** {dt_acc:.2f}")

    # Naive Bayes
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    nb_acc = accuracy_score(y_test, nb_model.predict(X_test))
    st.write(f"**Naive Bayes Accuracy:** {nb_acc:.2f}")

    # Gradient Boosting
    gb_model = GradientBoostingClassifier()
    gb_model.fit(X_train, y_train)
    gb_acc = accuracy_score(y_test, gb_model.predict(X_test))
    st.write(f"**Gradient Boosting Accuracy:** {gb_acc:.2f}")

    # Neural Network
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    nn_model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    nn_model.fit(X_train_scaled, y_train, epochs=10, verbose=0, batch_size=32)
    
    nn_acc = nn_model.evaluate(X_test_scaled, y_test, verbose=0)[1]
    st.write(f"**Neural Network Accuracy:** {nn_acc:.2f}")

    # Visualize Decision Tree
    st.write("### Decision Tree Visualization")
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_tree(dt_model, feature_names=X_train.columns, filled=True, ax=ax)
    st.pyplot(fig)

    # Correlation Heatmap
    st.write("### Correlation Heatmap")
    corr_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    st.pyplot(fig)
else:
    st.warning("Please upload a CSV file to proceed.")
