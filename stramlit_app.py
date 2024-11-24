import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from scipy.stats import mstats

# Set the page configuration
st.set_page_config(page_title="Loan Approval Analysis", layout="wide")

# Load the dataset
st.title("Loan Approval Prediction")
st.write("This application performs data cleaning, EDA, and various model predictions for a loan approval dataset.")

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.header("Dataset Overview")
    st.write(df.head())
    st.write(df.info())

    # Data Cleaning
    df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    df['Dependents'].replace('3+', '3', inplace=True)
    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median(), inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].median(), inplace=True)
    df['Dependents'] = pd.to_numeric(df['Dependents'], errors='coerce').fillna(0)

    # Outlier Handling
    columns_to_handle = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    for column in columns_to_handle:
        df[column] = mstats.winsorize(df[column], limits=[0.09, 0.09])

    st.header("Data After Cleaning")
    st.write(df.describe())

    # Exploratory Data Analysis
    st.header("Exploratory Data Analysis")
    if st.checkbox("Show Distribution of Applicant Income"):
        plt.hist(df['ApplicantIncome'], bins=10, edgecolor='black')
        plt.xlabel('Applicant Income')
        plt.ylabel('Frequency')
        st.pyplot(plt)

    if st.checkbox("Show Correlation Matrix"):
        correlation_matrix = df[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        st.pyplot(plt)

    # Encoding categorical variables
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
    df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
    df = pd.get_dummies(df, columns=['Property_Area'], drop_first=True)
    df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})

    # Split the dataset
    X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
    y = df['Loan_Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training
    st.header("Model Training and Prediction")

    model_type = st.selectbox("Select a model", ["Decision Tree", "Naive Bayes", "Gradient Boosting", "Neural Network"])

    if model_type == "Decision Tree":
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: {accuracy:.2f}")

        plt.figure(figsize=(15, 10))
        plot_tree(model, feature_names=X.columns, filled=True)
        st.pyplot(plt)

    elif model_type == "Naive Bayes":
        model = GaussianNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: {accuracy:.2f}")

    elif model_type == "Gradient Boosting":
        model = GradientBoostingClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: {accuracy:.2f}")

    elif model_type == "Neural Network":
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        nn_model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        nn_model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, verbose=1)
        loss, accuracy = nn_model.evaluate(X_test_scaled, y_test)
        st.write(f"Neural Network Test Accuracy: {accuracy:.2f}")
