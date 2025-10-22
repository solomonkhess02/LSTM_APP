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
# Random Seed Control
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
def load_any(file, purpose="Training"):
    """Load CSV or Excel with optional sheet selection."""
    if any(file.name.endswith(x) for x in [".xlsx", ".xls"]):
        excel_file = pd.ExcelFile(file)
        sheet_names = excel_file.sheet_names
        sheet_choice = st.selectbox(f"Select Sheet for {purpose} Data:", sheet_names, key=f"{purpose}_sheet")
        df = excel_file.parse(sheet_choice)
    else:
        df = pd.read_csv(file)
        st.caption(f"{purpose} data loaded from CSV")
    return df

def create_windows_multivariate(X, y, window):
    """Create sequences for LSTM."""
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i:i+window])
        ys.append(y[i+window])
    return np.array(Xs), np.array(ys)

# ----------------------
# Upload Training Data
# ----------------------
st.subheader("📂 Upload Training Data (CSV or Excel)")
train_file = st.file_uploader("Upload a CSV or Excel file for training", type=['csv', 'xlsx', 'xls'], key="train_file")

if train_file is not None:
    df_train = load_any(train_file, purpose="Training")
    st.write("📊 Training Data Preview")
    st.dataframe(df_train.head())

    # Column selection
    with st.expander("Column Selection", expanded=True):
        all_cols = list(df_train.columns)
        feature_cols = st.multiselect("Select Features (Inputs):", all_cols)
        target_col = st.selectbox("Select Target (Output):", [c for c in all_cols if c not in feature_cols])

    proceed = feature_cols and target_col
    if proceed:
        df_numeric = df_train[feature_cols + [target_col]].apply(pd.to_numeric, errors='coerce').dropna()
        if df_numeric.empty:
            st.error("❌ No numeric data after cleaning.")
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
                val_split = st.slider("Validation Split", 0.0, 0.4, 0.0, 0.05)

            # ----------------------
            # Layer Configuration
            # ----------------------
            st.subheader("🧠 LSTM Layer Configuration")
            layer_neurons, layer_dropouts = [], []
            for i in range(num_layers):
                neurons = st.slider(f"Neurons in Layer {i+1}", 1, 256, 64, 1, key=f"layer_{i+1}_neurons")
                dropout = st.slider(f"Dropout Rate (Layer {i+1})", 0.0, 0.5, 0.2, 0.05, key=f"layer_{i+1}_dropout")
                layer_neurons.append(neurons)
                layer_dropouts.append(dropout)
            # ----------------------  
            # Shape Debug Info (Testing)
            # ----------------------
            with st.expander("🔍 Data Shape Diagnostics (Test)", expanded=False):
                st.write(f"**Raw X_test:** {X_test.shape}")
                st.write(f"**Raw y_test:** {y_test.shape}")
                st.write(f"**Scaled X_test:** {X_test_scaled.shape}")
                st.write(f"**Scaled y_test:** {y_test_scaled.shape}")
                st.write(f"**Sequence X_test_seq:** {X_test_seq.shape}")
                st.write(f"**Sequence y_test_seq:** {y_test_seq.shape}")

            # ----------------------
            # Scale Data
            # ----------------------
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
            X_scaled = scaler_X.fit_transform(X)
            y_scaled = scaler_y.fit_transform(y)

            # Sequence windows
            X_seq, y_seq = create_windows_multivariate(X_scaled, y_scaled, window_size)
            # ----------------------  
            # Shape Debug Info (Training)
            # ----------------------
            with st.expander("🔍 Data Shape Diagnostics (Training)", expanded=False):
                st.write(f"**Raw X_train:** {X.shape}")
                st.write(f"**Raw y_train:** {y.shape}")
                st.write(f"**Scaled X_train:** {X_scaled.shape}")
                st.write(f"**Scaled y_train:** {y_scaled.shape}")
                st.write(f"**Sequence X_seq:** {X_seq.shape}")
                st.write(f"**Sequence y_seq:** {y_seq.shape}")
                st.write(f"**LSTM input shape used:** ({window_size}, {len(feature_cols)})")

            if X_seq.shape[0] == 0:
                st.error("❌ Window size too large for dataset.")
            else:
                if st.button("🚀 Train LSTM Model"):
                    n_features = len(feature_cols)

                    # Build model
                    model = Sequential()
                    for i in range(num_layers):
                        return_seq = (i < num_layers - 1)
                        model.add(LSTM(
                            layer_neurons[i],
                            activation=activation_fn,
                            return_sequences=return_seq,
                            input_shape=(window_size, n_features) if i == 0 else None
                        ))
                        if layer_dropouts[i] > 0:
                            model.add(Dropout(layer_dropouts[i]))
                    model.add(Dense(1))

                    # Optimizer
                    if optimizer_choice == "Adam":
                        optimizer = Adam(learning_rate=learning_rate)
                    elif optimizer_choice == "RMSprop":
                        optimizer = RMSprop(learning_rate=learning_rate)
                    else:
                        optimizer = Nadam(learning_rate=learning_rate)

                    model.compile(loss=loss_fn, optimizer=optimizer)

                    # ----------------------  
                    # Shape Debug Info (Testing)
                    # ----------------------
                    with st.expander("🔍 Data Shape Diagnostics (Test)", expanded=False):
                        st.write(f"**Raw X_test:** {X_test.shape}")
                        st.write(f"**Raw y_test:** {y_test.shape}")
                        st.write(f"**Scaled X_test:** {X_test_scaled.shape}")
                        st.write(f"**Scaled y_test:** {y_test_scaled.shape}")
                        st.write(f"**Sequence X_test_seq:** {X_test_seq.shape}")
                        st.write(f"**Sequence y_test_seq:** {y_test_seq.shape}")

                    # Train
                    history = model.fit(
                        X_seq, y_seq,
                        epochs=num_epochs,
                        batch_size=batch_size,
                        validation_split=val_split,
                        shuffle=False,
                        verbose=1
                    )
                    # ----------------------  
                    # Shape Debug Info (Testing)
                    # ----------------------
                    with st.expander("🔍 Data Shape Diagnostics (Test)", expanded=False):
                        st.write(f"**Raw X_test:** {X_test.shape}")
                        st.write(f"**Raw y_test:** {y_test.shape}")
                        st.write(f"**Scaled X_test:** {X_test_scaled.shape}")
                        st.write(f"**Scaled y_test:** {y_test_scaled.shape}")
                        st.write(f"**Sequence X_test_seq:** {X_test_seq.shape}")
                        st.write(f"**Sequence y_test_seq:** {y_test_seq.shape}")

                    # Store model & scalers in session_state
                    st.session_state['model'] = model
                    st.session_state['scaler_X'] = scaler_X
                    st.session_state['scaler_y'] = scaler_y
                    st.session_state['window_size'] = window_size
                    st.session_state['feature_cols'] = feature_cols
                    st.session_state['target_col'] = target_col

                    # Full dataset prediction
                    y_pred = model.predict(X_seq, verbose=0)
                    # y_true_inv = scaler_y.inverse_transform(y_seq.reshape(-1, 1))
                    y_true_inv = scaler_y.inverse_transform(y_seq) # No reshape needed now
                    y_pred_inv = scaler_y.inverse_transform(y_pred)
                    r2_full = r2_score(y_true_inv, y_pred_inv)
                    st.success(f"✅ Training R² = {r2_full:.4f}")

                    # Plots
                    fig1, ax1 = plt.subplots()
                    ax1.plot(y_true_inv, label='Actual')
                    ax1.plot(y_pred_inv, label='Predicted', alpha=0.7)
                    ax1.legend()
                    ax1.set_title("Predicted vs Actual (Training)")
                    st.pyplot(fig1)

                    fig2, ax2 = plt.subplots()
                    ax2.scatter(y_true_inv, y_pred_inv, alpha=0.6, color='purple')
                    lim_min = min(y_true_inv.min(), y_pred_inv.min())
                    lim_max = max(y_true_inv.max(), y_pred_inv.max())
                    ax2.plot([lim_min, lim_max], [lim_min, lim_max], 'r--')
                    ax2.set_xlabel("Actual")
                    ax2.set_ylabel("Predicted")
                    ax2.set_title("Parity Plot (Training)")
                    st.pyplot(fig2)

                    fig3, ax3 = plt.subplots()
                    ax3.plot(history.history['loss'], label='Training Loss')
                    if val_split > 0 and 'val_loss' in history.history:
                        ax3.plot(history.history['val_loss'], label='Validation Loss')
                    ax3.set_xlabel("Epochs")
                    ax3.set_ylabel("Loss")
                    ax3.legend()
                    ax3.set_title("Loss Curve")
                    st.pyplot(fig3)

# ----------------------
# Upload Test Data
# ----------------------
st.subheader("📂 Upload Test Data (CSV or Excel)")
test_file = st.file_uploader("Upload a CSV or Excel file for testing", type=['csv', 'xlsx', 'xls'], key="test_file")

if test_file is not None:
    if 'model' not in st.session_state:
        st.warning("⚠️ Train the model first before testing.")
    else:
        df_test = load_any(test_file, purpose="Test")
        st.write("📊 Test Data Preview")
        st.dataframe(df_test.head())

        # Select only features used in training
        feature_cols = st.session_state['feature_cols']
        target_col = st.session_state['target_col']

        missing_features = [f for f in feature_cols if f not in df_test.columns]
        if missing_features:
            st.error(f"❌ Test data missing these required features: {missing_features}")
        else:
            df_test_numeric = df_test[feature_cols + [target_col]].apply(pd.to_numeric, errors='coerce').dropna()
            if df_test_numeric.empty:
                st.error("❌ No numeric test data after cleaning.")
            else:
                X_test = df_test_numeric[feature_cols].values
                y_test = df_test_numeric[[target_col]].values

                # Scale using training scalers
                X_test_scaled = st.session_state['scaler_X'].transform(X_test)
                y_test_scaled = st.session_state['scaler_y'].transform(y_test)

                # Create test sequences
                X_test_seq, y_test_seq = create_windows_multivariate(X_test_scaled, y_test_scaled, st.session_state['window_size'])

                if X_test_seq.shape[0] == 0:
                    st.error("❌ Window size too large for test data.")
                else:
                    if st.button("📊 Predict on Test Data"):
                        y_test_pred_scaled = st.session_state['model'].predict(X_test_seq, verbose=0)
                        y_test_true_inv = st.session_state['scaler_y'].inverse_transform(y_test_seq.reshape(-1,1))
                        y_test_pred_inv = st.session_state['scaler_y'].inverse_transform(y_test_pred_scaled)

                        r2_test = r2_score(y_test_true_inv, y_test_pred_inv)
                        st.success(f"✅ Test R² = {r2_test:.4f}")

                        # Plots
                        fig_t1, ax_t1 = plt.subplots()
                        ax_t1.plot(y_test_true_inv, label='Actual')
                        ax_t1.plot(y_test_pred_inv, label='Predicted', alpha=0.7)
                        ax_t1.legend()
                        ax_t1.set_title("Predicted vs Actual (Test)")
                        st.pyplot(fig_t1)

                        fig_t2, ax_t2 = plt.subplots()
                        ax_t2.scatter(y_test_true_inv, y_test_pred_inv, alpha=0.6, color='purple')
                        lim_min = min(y_test_true_inv.min(), y_test_pred_inv.min())
                        lim_max = max(y_test_true_inv.max(), y_test_pred_inv.max())
                        ax_t2.plot([lim_min, lim_max], [lim_min, lim_max], 'r--')
                        ax_t2.set_xlabel("Actual")
                        ax_t2.set_ylabel("Predicted")
                        ax_t2.set_title("Parity Plot (Test)")
                        st.pyplot(fig_t2)



