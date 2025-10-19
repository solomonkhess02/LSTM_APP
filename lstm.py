import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random, os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

# ----------------------
# Page config
# ----------------------
st.set_page_config(page_title="LSTM Time Series Demo (Multi-feature)", layout="wide")
st.title("📈 LSTM Time Series Demo (Multi-feature)")

# ----------------------
# Set random seed
# ----------------------
st.sidebar.subheader("🧮 Random Seed Control")
seed_val = st.sidebar.number_input("Set Random Seed", min_value=0, max_value=9999, value=42, step=1)

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

set_random_seed(seed_val)

# ----------------------
# Helper functions
# ----------------------
def load_any(main_file, purpose="Training"):
    if any(main_file.name.endswith(x) for x in [".xlsx", ".xls"]):
        excel_file = pd.ExcelFile(main_file)
        sheet_names = excel_file.sheet_names
        sheet_choice = st.selectbox(f"Select Sheet for {purpose} Data:", sheet_names, key=f"{purpose}_sheet")
        df = excel_file.parse(sheet_choice)
    else:
        df = pd.read_csv(main_file)
        st.caption(f"{purpose} data loaded from CSV")
    return df

def create_windows_multivariate(X, y, window):
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i:i+window, :])
        ys.append(y[i+window, 0])
    return np.array(Xs), np.array(ys)

# ----------------------
# Upload training data
# ----------------------
st.subheader("📂 Upload Training Data (CSV or Excel)")
main_file = st.file_uploader("Upload a CSV or Excel file for training", type=['csv', 'xlsx', 'xls'])

if main_file is not None:
    df_series = load_any(main_file, purpose="Training")
    st.write("📊 Training Data Preview")
    st.dataframe(df_series.head())

    # Column selection
    with st.expander("Column Selection", expanded=True):
        all_cols = list(df_series.columns)
        feature_cols = st.multiselect("Select Features (Inputs):", all_cols)
        target_col = st.selectbox("Select Target (Output):", [c for c in all_cols if c not in feature_cols])

    proceed = feature_cols and target_col
    if proceed:
        df_numeric = df_series[feature_cols + [target_col]].apply(pd.to_numeric, errors='coerce').dropna()

        if df_numeric.empty:
            st.error("❌ After numeric conversion and NA dropping, no data remains.")
        else:
            X = df_numeric[feature_cols].values
            y = df_numeric[[target_col]].values

            # ----------------------
            # Hyperparameters
            # ----------------------
            st.subheader("⚙️ LSTM Hyperparameters")
            c1, c2, c3 = st.columns(3)
            with c1:
                window_size = st.slider("Window Size (timesteps)", 5, 200, 20, 1)
                num_layers = st.slider("Number of LSTM Layers", 1, 3, 1)
                loss_fn = st.selectbox("Loss Function", ["mse", "mae"])
            with c2:
                activation_fn = st.selectbox("Activation Function", ["tanh", "relu", "sigmoid"])
                optimizer_choice = st.selectbox("Optimizer", ["Adam", "RMSprop", "Nadam"])
                learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001, 0.0001, format="%.4f")
            with c3:
                batch_size = st.slider("Batch Size", 8, 256, 32, 1)
                num_epochs = st.slider("Epochs", 5, 300, 30, 5)
                val_split = st.slider("Validation Split (optional)", 0.0, 0.4, 0.0, 0.05)

            # ----------------------
            # Layer Config
            # ----------------------
            st.subheader("🧠 LSTM Layer Configuration")
            layer_neurons, layer_dropouts = [], []
            for i in range(num_layers):
                neurons = st.slider(f"Neurons in Layer {i+1}", 1, 256, 64, 1, key=f"layer_{i+1}_neurons")
                dropout = st.slider(f"Dropout Rate (Layer {i+1})", 0.0, 0.5, 0.2, 0.05, key=f"layer_{i+1}_dropout")
                layer_neurons.append(neurons)
                layer_dropouts.append(dropout)

            # ----------------------
            # Scaling (fit on full data)
            # ----------------------
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
            scaled_X = scaler_X.fit_transform(X)
            scaled_y = scaler_y.fit_transform(y)

            # ----------------------
            # Create sequence windows (no test split)
            # ----------------------
            X_seq, y_seq = create_windows_multivariate(scaled_X, scaled_y, window_size)

            if X_seq.shape[0] == 0:
                st.error("❌ Window size too large relative to dataset length.")
            else:
                if st.button("🚀 Train LSTM Model"):
                    n_features = len(feature_cols)

                    # Build model
                    model = Sequential()
                    for i in range(num_layers):
                        return_sequences = (i < num_layers - 1)
                        model.add(LSTM(
                            layer_neurons[i],
                            activation=activation_fn,
                            return_sequences=return_sequences,
                            input_shape=(window_size, n_features) if i == 0 else None
                        ))
                        if layer_dropouts[i] > 0:
                            model.add(Dropout(layer_dropouts[i]))

                    model.add(Dense(1))

                    # Optimizer setup
                    if optimizer_choice == "Adam":
                        optimizer = Adam(learning_rate=learning_rate)
                    elif optimizer_choice == "RMSprop":
                        optimizer = RMSprop(learning_rate=learning_rate)
                    else:
                        optimizer = Nadam(learning_rate=learning_rate)

                    model.compile(loss=loss_fn, optimizer=optimizer)

                    # Train (no early stopping, no shuffling)
                    history = model.fit(
                        X_seq, y_seq,
                        epochs=num_epochs,
                        batch_size=batch_size,
                        verbose=1,
                        validation_split=val_split,
                        shuffle=False
                    )

                    # Predictions on the full dataset
                    y_pred = model.predict(X_seq, verbose=0)

                    y_true_inv = scaler_y.inverse_transform(y_seq.reshape(-1, 1))
                    y_pred_inv = scaler_y.inverse_transform(y_pred)

                    r2_full = r2_score(y_true_inv, y_pred_inv)
                    st.success(f"✅ Model trained! Full-data R² = {r2_full:.4f}")

                    # ----------------------
                    # Plots
                    # ----------------------
                    fig1, ax1 = plt.subplots()
                    ax1.plot(y_true_inv, label='Actual', color='blue')
                    ax1.plot(y_pred_inv, label='Predicted', color='red', alpha=0.7)
                    ax1.legend()
                    ax1.set_title("Predicted vs Actual (Full Dataset)")
                    st.pyplot(fig1)

                    fig2, ax2 = plt.subplots()
                    ax2.scatter(y_true_inv, y_pred_inv, alpha=0.6, color='purple')
                    lim_min = float(min(y_true_inv.min(), y_pred_inv.min()))
                    lim_max = float(max(y_true_inv.max(), y_pred_inv.max()))
                    ax2.plot([lim_min, lim_max], [lim_min, lim_max], 'r--')
                    ax2.set_xlabel("Actual")
                    ax2.set_ylabel("Predicted")
                    ax2.set_title("Parity Plot (Full Data)")
                    st.pyplot(fig2)

                    fig3, ax3 = plt.subplots()
                    ax3.plot(history.history['loss'], label="Training Loss")
                    if 'val_loss' in history.history and val_split > 0:
                        ax3.plot(history.history['val_loss'], label="Validation Loss")
                    ax3.set_title(f"Training vs Validation Loss ({loss_fn.upper()})")
                    ax3.set_xlabel("Epochs")
                    ax3.set_ylabel("Loss")
                    ax3.legend()
                    st.pyplot(fig3)

            # ----------------------
            # Upload test data
            # ----------------------
            st.subheader("📂 Upload Test Data (CSV or Excel)")
            test_file = st.file_uploader("Upload a CSV or Excel file for testing", type=['csv', 'xlsx', 'xls'], key="test_file")
            
            if test_file is not None and proceed:
                df_test = load_any(test_file, purpose="Test")
                st.write("📊 Test Data Preview")
                st.dataframe(df_test.head())
            
                # Ensure columns match training features + target
                if all(col in df_test.columns for col in feature_cols + [target_col]):
                    df_test_numeric = df_test[feature_cols + [target_col]].apply(pd.to_numeric, errors='coerce').dropna()
            
                    if df_test_numeric.empty:
                        st.error("❌ After numeric conversion and NA dropping, no test data remains.")
                    else:
                        X_test = df_test_numeric[feature_cols].values
                        y_test = df_test_numeric[[target_col]].values
            
                        # Scale using training data scalers
                        X_test_scaled = scaler_X.transform(X_test)
                        y_test_scaled = scaler_y.transform(y_test)
            
                        # Create sequence windows for test
                        X_test_seq, y_test_seq = create_windows_multivariate(X_test_scaled, y_test_scaled, window_size)
            
                        if X_test_seq.shape[0] == 0:
                            st.error("❌ Window size too large relative to test dataset length.")
                        else:
                            if st.button("📊 Predict on Test Data"):
                                y_test_pred_scaled = model.predict(X_test_seq, verbose=0)
                                y_test_true_inv = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1))
                                y_test_pred_inv = scaler_y.inverse_transform(y_test_pred_scaled)
            
                                r2_test = r2_score(y_test_true_inv, y_test_pred_inv)
                                st.success(f"✅ Test R² = {r2_test:.4f}")
            
                                # ----------------------
                                # Test Plots
                                # ----------------------
                                fig_test1, ax_test1 = plt.subplots()
                                ax_test1.plot(y_test_true_inv, label='Actual', color='blue')
                                ax_test1.plot(y_test_pred_inv, label='Predicted', color='red', alpha=0.7)
                                ax_test1.legend()
                                ax_test1.set_title("Predicted vs Actual (Test Data)")
                                st.pyplot(fig_test1)
            
                                fig_test2, ax_test2 = plt.subplots()
                                ax_test2.scatter(y_test_true_inv, y_test_pred_inv, alpha=0.6, color='purple')
                                lim_min = float(min(y_test_true_inv.min(), y_test_pred_inv.min()))
                                lim_max = float(max(y_test_true_inv.max(), y_test_pred_inv.max()))
                                ax_test2.plot([lim_min, lim_max], [lim_min, lim_max], 'r--')
                                ax_test2.set_xlabel("Actual")
                                ax_test2.set_ylabel("Predicted")
                                ax_test2.set_title("Parity Plot (Test Data)")
                                st.pyplot(fig_test2)
                else:
                    st.error("❌ Test data does not contain all selected training features + target column.")
