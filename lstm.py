import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

st.title("📈 LSTM Time Series Demo")

# ----------------------
# Upload main dataset
# ----------------------
st.subheader("📂 Upload Training CSV Data")
main_file = st.file_uploader("Upload a CSV file with a 'value' column", type=['csv'])

if main_file is not None:
    df_series = pd.read_csv(main_file)
    st.write("Training Data Preview", df_series.head())

    # Plot input series before training
    fig, ax = plt.subplots()
    ax.plot(df_series['value'], label='Input Series', color='blue')
    ax.set_title("Raw Input Time Series")
    ax.legend()
    st.pyplot(fig)

    # Select window size and number of neurons
    window_size = st.slider("Window Size", min_value=5, max_value=50, value=20)
    num_neurons = st.slider("Number of LSTM Neurons", min_value=10, max_value=200, value=50, step=10)

    # Scale the full series
    scaler = MinMaxScaler()
    scaled_series = scaler.fit_transform(df_series[['value']].values)

    # Split train/test (80/20)
    train_size = int(len(scaled_series) * 0.8)
    train, test = scaled_series[:train_size], scaled_series[train_size:]

    # Function to build windows
    def create_windows(series, window):
        X, y = [], []
        for i in range(len(series) - window):
            X.append(series[i:i+window])
            y.append(series[i+window])
        return np.array(X), np.array(y)

    # Build train and test sets
    X_train, y_train = create_windows(train, window_size)
    X_test, y_test = create_windows(test, window_size)

    # ----------------------
    # Train model
    # ----------------------
    if st.button("Train LSTM Model"):
        if X_train.shape[0] == 0 or X_test.shape[0] == 0:
            st.error("❌ Window size is too large for the dataset. Reduce the slider value.")
        else:
            # Reshape for LSTM
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

            # Build & train model with dynamic neurons
            model = Sequential()
            model.add(LSTM(num_neurons, input_shape=(window_size, 1)))
            model.add(Dense(1))
            model.compile(loss='mse', optimizer='adam')
            model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)

            # Save model + scaler to session state ✅
            st.session_state["model"] = model
            st.session_state["scaler"] = scaler
            st.session_state["window_size"] = window_size

            # Predict on internal test set
            y_pred = model.predict(X_test)
            y_true_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
            y_pred_inv = scaler.inverse_transform(y_pred)

            # R² score
            r2 = r2_score(y_true_inv, y_pred_inv)
            st.success(f"✅ Model trained! R² Score = {r2:.4f}")

            # Plot results
            fig1, ax1 = plt.subplots()
            ax1.plot(y_true_inv, label='Actual (Test)', color='orange')
            ax1.plot(y_pred_inv, label='LSTM Prediction', color='red')
            ax1.legend()
            ax1.set_title("Prediction vs Actual (Internal Test)")
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots()
            ax2.scatter(y_true_inv, y_pred_inv, alpha=0.6, color='green')
            ax2.plot([y_true_inv.min(), y_true_inv.max()],
                     [y_true_inv.min(), y_true_inv.max()], 'r--')
            ax2.set_xlabel("Actual Values")
            ax2.set_ylabel("Predicted Values")
            ax2.set_title("Parity Plot (Internal Test)")
            st.pyplot(fig2)

    # ----------------------
    # External Test CSV
    # ----------------------
    st.subheader("📂 Test on External CSV Data")
    uploaded_file = st.file_uploader("Upload a separate CSV file with a 'value' column", type=['csv'], key="external")

    if uploaded_file is not None:
        if "model" not in st.session_state:
            st.warning("⚠️ Train the model first before testing on external data.")
        else:
            df_ext = pd.read_csv(uploaded_file)
            st.write("External Test Data Preview", df_ext.head())

            # Retrieve trained model + scaler
            scaler = st.session_state["scaler"]
            model = st.session_state["model"]
            window_size = st.session_state["window_size"]

            # Scale and create windows
            scaled_ext = scaler.transform(df_ext[['value']].values)
            X_ext, y_ext = create_windows(scaled_ext, window_size)

            if X_ext.shape[0] > 0:
                X_ext = X_ext.reshape((X_ext.shape[0], X_ext.shape[1], 1))
                y_ext_pred = model.predict(X_ext)

                # Inverse transform
                y_ext_true = scaler.inverse_transform(y_ext.reshape(-1, 1))
                y_ext_pred_inv = scaler.inverse_transform(y_ext_pred)

                # R² score
                r2_ext = r2_score(y_ext_true, y_ext_pred_inv)
                st.success(f"📊 External Test R² Score = {r2_ext:.4f}")

                # Plot results
                fig3, ax3 = plt.subplots()
                ax3.plot(y_ext_true, label='Actual (External)', color='purple')
                ax3.plot(y_ext_pred_inv, label='Prediction (External)', color='cyan')
                ax3.legend()
                ax3.set_title("Prediction vs Actual (External Test)")
                st.pyplot(fig3)

                fig4, ax4 = plt.subplots()
                ax4.scatter(y_ext_true, y_ext_pred_inv, alpha=0.6, color='brown')
                ax4.plot([y_ext_true.min(), y_ext_true.max()],
                         [y_ext_true.min(), y_ext_true.max()], 'r--')
                ax4.set_xlabel("Actual Values")
                ax4.set_ylabel("Predicted Values")
                ax4.set_title("Parity Plot (External Test)")
                st.pyplot(fig4)
            else:
                st.error("❌ Window size too large for external dataset.")

    # Debug info
    st.write(f"Train shape: X={X_train.shape}, y={y_train.shape}")
    st.write(f"Test shape: X={X_test.shape}, y={y_test.shape}")

else:
    st.info("👆 Please upload a CSV file with a 'value' column to start.")




