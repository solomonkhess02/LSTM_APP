import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

st.set_page_config(page_title="LSTM Time Series Demo (Multi-feature)", layout="wide")
st.title("📈 LSTM Time Series Demo (Multi-feature)")

# ----------------------
# Helpers
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
# Upload main dataset
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
        feature_cols = st.multiselect(
            "Select Features (Input Columns):",
            options=all_cols,
            help="Select one or more input variables used to predict the target.",
        )
        target_col = st.selectbox(
            "Select Target (Output Column):",
            options=[c for c in all_cols if c not in feature_cols],
            help="Select the output variable to predict.",
        )

    proceed = feature_cols and target_col
    if proceed:
        # Clean numeric casting
        df_numeric = df_series[feature_cols + [target_col]].copy()
        df_numeric = df_numeric.apply(pd.to_numeric, errors='coerce')
        df_numeric = df_numeric.dropna()

        if df_numeric.empty:
            st.error("❌ After numeric conversion and NA dropping, no data remains. Please check selected columns.")
        else:
            series_X = df_numeric[feature_cols].values
            series_y = df_numeric[[target_col]].values

            # Plot first selected feature for quick sanity check
            with st.expander("Input Signal Preview"):
                fig, ax = plt.subplots()
                if len(series_X) > 2000:
                    ax.plot(series_X[::10, 0], label=f"Sampled Feature: {feature_cols[0]}", color='blue')
                else:
                    ax.plot(series_X[:, 0], label=f"Feature: {feature_cols[0]}", color='blue')
                ax.set_title(f"First Feature Time Series ({feature_cols[0]})")
                ax.legend()
                st.pyplot(fig)

            # ----------------------
            # Hyperparameters
            # ----------------------
            st.subheader("⚙️ LSTM Hyperparameters")
            c1, c2, c3 = st.columns(3)
            with c1:
                window_size = st.slider("Window Size (timesteps)", min_value=5, max_value=200, value=20, step=1)
                num_layers = st.slider("Number of LSTM Layers", min_value=1, max_value=3, value=1)
                loss_fn = st.selectbox("Loss Function", ["mse", "mae"])
            with c2:
                num_neurons = st.slider("LSTM Units per Layer", min_value=10, max_value=256, value=64, step=8)
                activation_fn = st.selectbox("Activation Function", ["tanh", "relu", "sigmoid"])
                batch_size = st.slider("Batch Size", min_value=8, max_value=256, value=32, step=8)
            with c3:
                learning_rate = st.slider("Learning Rate", min_value=0.0001, max_value=0.01, value=0.001, step=0.0001, format="%.4f")
                num_epochs = st.slider("Epochs", min_value=5, max_value=300, value=30, step=5)
                val_split = st.slider("Validation Split", min_value=0.0, max_value=0.4, value=0.2, step=0.05)

            # ----------------------
            # Scaling
            # ----------------------
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
            scaled_X = scaler_X.fit_transform(series_X)
            scaled_y = scaler_y.fit_transform(series_y)

            # ----------------------
            # Train/Test split and windowing
            # ----------------------
            train_size = int(len(scaled_X) * 0.8)
            train_X, test_X = scaled_X[:train_size], scaled_X[train_size:]
            train_y, test_y = scaled_y[:train_size], scaled_y[train_size:]

            X_train, y_train = create_windows_multivariate(train_X, train_y, window_size)
            X_test, y_test = create_windows_multivariate(test_X, test_y, window_size)

            # Safety checks
            if X_train.shape[0] == 0 or X_test.shape[0] == 0:
                st.error("❌ Window size is too large relative to dataset length. Reduce the window size.")
            else:
                # ----------------------
                # Train model
                # ----------------------
                if st.button("Train LSTM Model"):
                    # Reshape to (batch, timesteps, features)
                    n_features = len(feature_cols)
                    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
                    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))

                    model = Sequential()
                    for i in range(num_layers):
                        return_sequences = (i < num_layers - 1)
                        model.add(
                            LSTM(
                                num_neurons,
                                activation=activation_fn,
                                return_sequences=return_sequences,
                                input_shape=(window_size, n_features) if i == 0 else None
                            )
                        )
                    model.add(Dense(1))

                    optimizer = Adam(learning_rate=learning_rate)
                    model.compile(loss=loss_fn, optimizer=optimizer)

                    history = model.fit(
                        X_train, y_train,
                        epochs=num_epochs,
                        batch_size=batch_size,
                        verbose=0,
                        validation_split=val_split,
                    )

                    # Save session state
                    st.session_state["model"] = model
                    st.session_state["scaler_X"] = scaler_X
                    st.session_state["scaler_y"] = scaler_y
                    st.session_state["window_size"] = window_size
                    st.session_state["feature_cols"] = feature_cols
                    st.session_state["target_col"] = target_col

                    # Evaluate on internal test
                    y_pred_test = model.predict(X_test, verbose=0)
                    y_pred_train = model.predict(X_train, verbose=0)

                    y_pred_test_inv = scaler_y.inverse_transform(y_pred_test)
                    y_pred_train_inv = scaler_y.inverse_transform(y_pred_train)
                    y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1))
                    y_train_inv = scaler_y.inverse_transform(y_train.reshape(-1, 1))

                    r2_train = r2_score(y_train_inv, y_pred_train_inv)
                    r2_test = r2_score(y_test_inv, y_pred_test_inv)
                    st.success(f"✅ Model trained! Train R² = {r2_train:.4f}, Test R² = {r2_test:.4f}")

                    # Plots
                    colA, colB = st.columns(2)

                    with colA:
                        fig1, ax1 = plt.subplots()
                        ax1.plot(y_test_inv, label='Actual (Test)', color='orange')
                        ax1.plot(y_pred_test_inv, label='LSTM Prediction', color='red')
                        ax1.legend()
                        ax1.set_title("Prediction vs Actual (Internal Test)")
                        st.pyplot(fig1)

                    with colB:
                        fig2, ax2 = plt.subplots()
                        ax2.scatter(y_test_inv, y_pred_test_inv, alpha=0.6, color='green')
                        lim_min = float(min(y_test_inv.min(), y_pred_test_inv.min()))
                        lim_max = float(max(y_test_inv.max(), y_pred_test_inv.max()))
                        ax2.plot([lim_min, lim_max], [lim_min, lim_max], 'r--')
                        ax2.set_xlabel("Actual Values")
                        ax2.set_ylabel("Predicted Values")
                        ax2.set_title("Parity Plot (Internal Test)")
                        st.pyplot(fig2)

                    fig3, ax3 = plt.subplots()
                    ax3.plot(history.history['loss'], label="Training Loss")
                    if 'val_loss' in history.history:
                        ax3.plot(history.history['val_loss'], label="Validation Loss")
                    ax3.set_title(f"Training vs Validation Loss ({loss_fn.upper()})")
                    ax3.set_xlabel("Epochs")
                    ax3.set_ylabel("Loss")
                    ax3.legend()
                    st.pyplot(fig3)

                    train_errors = y_train_inv.flatten() - y_pred_train_inv.flatten()
                    test_errors = y_test_inv.flatten() - y_pred_test_inv.flatten()

                    fig4, ax4 = plt.subplots()
                    ax4.hist(train_errors, bins=30, alpha=0.7, label="Train Errors", color="blue")
                    ax4.hist(test_errors, bins=30, alpha=0.7, label="Test Errors", color="red")
                    ax4.set_title("Error Distribution (Train vs Test)")
                    ax4.set_xlabel("Error")
                    ax4.set_ylabel("Frequency")
                    ax4.legend()
                    st.pyplot(fig4)

        # ----------------------
        # ----------------------
        # External Test Dataset
        # ----------------------
        st.subheader("📂 Upload External Test Data (Optional)")
        test_file = st.file_uploader("Upload a separate CSV or Excel file for testing", type=['csv', 'xlsx', 'xls'])
        
        if test_file is not None and "model" in st.session_state:
            # Allow user to select sheet if Excel
            df_test = load_any(test_file, purpose="External Test")
            st.write("📊 External Test Data Preview")
            st.dataframe(df_test.head())
        
            feat_cols = st.session_state["feature_cols"]
            target_col = st.session_state["target_col"]
        
            # Check for missing required columns
            missing = [c for c in feat_cols + [target_col] if c not in df_test.columns]
            if missing:
                st.error(f"❌ Missing required columns in external test file: {missing}")
            else:
                df_test_num = df_test[feat_cols + [target_col]].copy()
                df_test_num = df_test_num.apply(pd.to_numeric, errors='coerce').dropna()
                if df_test_num.empty:
                    st.error("❌ External test data has no valid numeric rows after cleaning.")
                else:
                    # Prepare data
                    X_ext = df_test_num[feat_cols].values
                    y_ext = df_test_num[[target_col]].values
        
                    # Load trained scalers + model
                    scaler_X = st.session_state["scaler_X"]
                    scaler_y = st.session_state["scaler_y"]
                    window_size = st.session_state["window_size"]
                    model = st.session_state["model"]
        
                    # 🔹 Use the same trained scalers (NO refitting)
                    scaled_X_ext = scaler_X.transform(X_ext)
                    scaled_y_ext = scaler_y.transform(y_ext)
        
                    # Create sliding windows
                    X_ext_win, y_ext_win = create_windows_multivariate(scaled_X_ext, scaled_y_ext, window_size)
        
                    if X_ext_win.shape[0] == 0:
                        st.error("❌ Window size too large for external test file length.")
                    else:
                        n_features = len(feat_cols)
                        X_ext_win = X_ext_win.reshape((X_ext_win.shape[0], X_ext_win.shape[1], n_features))
        
                        # Predict
                        y_pred_ext = model.predict(X_ext_win, verbose=0)
        
                        # Inverse scaling
                        y_ext_inv = scaler_y.inverse_transform(y_ext_win.reshape(-1, 1))
                        y_pred_ext_inv = scaler_y.inverse_transform(y_pred_ext)
        
                        # R² score
                        r2_ext = r2_score(y_ext_inv, y_pred_ext_inv)
                        st.success(f"📊 External Test R² = {r2_ext:.4f}")
        
                        # Plot results
                        fig5, ax5 = plt.subplots()
                        ax5.plot(y_ext_inv, label="Actual (External Test)", color="blue")
                        ax5.plot(y_pred_ext_inv, label="Prediction", color="red")
                        ax5.legend()
                        ax5.set_title("Prediction vs Actual (External Test)")
                        st.pyplot(fig5)
