import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Sample data generation
def generate_data(num_samples=1000):
    X = np.random.rand(num_samples, 10)  # 10 features
    y = (np.sum(X, axis=1) > 5).astype(int)  # Binary target
    return X, y

# Load data
X, y = generate_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the neural network model
nn_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
nn_model.fit(X_train_scaled, y_train, epochs=10, verbose=0, batch_size=32)

# Evaluate the model
nn_acc = nn_model.evaluate(X_test_scaled, y_test, verbose=0)[1]

# Streamlit app layout
st.title("Neural Network Example")
st.write(f"**Neural Network Accuracy:** {nn_acc:.2f}")

# Optional: Visualize some data
st.subheader("Sample Data Visualization")
df = pd.DataFrame(X, columns=[f"Feature {i+1}" for i in range(X.shape[1])])
sns.pairplot(df)
st.pyplot(plt)
