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
                batch_size = st.slider("Batch Size", 8, 256, 32, 8)
                learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001, 0.0001, format="%.4f")
            with c3:
                num_epochs = st.slider("Epochs", 5, 300, 30, 5)
                val_split = st.slider("Validation Split", 0.0, 0.4, 0.2, 0.05)

            # ----------------------
            # Per-layer neuron selection
            # ----------------------
            st.subheader("🧠 LSTM Layer Configuration")
            layer_neurons = []
            for i in range(num_layers):
                layer_neurons.append(
                    st.slider(f"Neurons in Layer {i+1}", 8, 256, 64, 8, key=f"layer{i+1}_neurons")
                )

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

            if X_train.shape[0] == 0 or X_test.shape[0] == 0:
                st.error("❌ Window size too large relative to dataset length. Reduce the window size.")
            else:
                # ----------------------
                # Train model
                # ----------------------
                if st.button("🚀 Train LSTM Model"):
                    n_features = len(feature_cols)
                    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
                    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))

                    model = Sequential()
                    for i in range(num_layers):
                        return_sequences = (i < num_layers - 1)
                        neurons = layer_neurons[i]
                        model.add(
                            LSTM(
                                neurons,
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
                    st.session_state.update({
                        "model": model,
                        "scaler_X": scaler_X,
                        "scaler_y": scaler_y,
                        "window_size": window_size,
                        "feature_cols": feature_cols,
                        "target_col": target_col
                    })

                    # ----------------------
                    # Predictions
                    # ----------------------
                    y_pred_train = model.predict(X_train, verbose=0)
                    y_pred_test = model.predict(X_test, verbose=0)

                    y_train_inv = scaler_y.inverse_transform(y_train.reshape(-1, 1))
                    y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1))
                    y_pred_train_inv = scaler_y.inverse_transform(y_pred_train)
                    y_pred_test_inv = scaler_y.inverse_transform(y_pred_test)

                    r2_train = r2_score(y_train_inv, y_pred_train_inv)
                    r2_test = r2_score(y_test_inv, y_pred_test_inv)
                    st.success(f"✅ Model trained! Train R² = {r2_train:.4f}, Test R² = {r2_test:.4f}")

                    # ----------------------
                    # Plots
                    # ----------------------
                    colA, colB = st.columns(2)

                    # (1) Test prediction vs actual
                    with colA:
                        fig1, ax1 = plt.subplots()
                        ax1.plot(y_test_inv, label='Actual (Test)', color='orange')
                        ax1.plot(y_pred_test_inv, label='Prediction', color='red')
                        ax1.legend()
                        ax1.set_title("Prediction vs Actual (Internal Test)")
                        st.pyplot(fig1)

                    # (2) Test parity
                    with colB:
                        fig2, ax2 = plt.subplots()
                        ax2.scatter(y_test_inv, y_pred_test_inv, alpha=0.6, color='green')
                        lim_min = float(min(y_test_inv.min(), y_pred_test_inv.min()))
                        lim_max = float(max(y_test_inv.max(), y_pred_test_inv.max()))
                        ax2.plot([lim_min, lim_max], [lim_min, lim_max], 'r--')
                        ax2.set_xlabel("Actual Values")
                        ax2.set_ylabel("Predicted Values")
                        ax2.set_title("Parity Plot (Test Data)")
                        st.pyplot(fig2)

                    # (3) Train prediction vs actual vs time
                    fig3, ax3 = plt.subplots()
                    ax3.plot(y_train_inv, label='Actual (Train)', color='blue')
                    ax3.plot(y_pred_train_inv, label='Prediction (Train)', color='red', alpha=0.7)
                    ax3.set_title("Predicted vs Actual vs Time (Training Data)")
                    ax3.legend()
                    st.pyplot(fig3)

                    # (4) Train parity
                    fig4, ax4 = plt.subplots()
                    ax4.scatter(y_train_inv, y_pred_train_inv, alpha=0.6, color='purple')
                    lim_min = float(min(y_train_inv.min(), y_pred_train_inv.min()))
                    lim_max = float(max(y_train_inv.max(), y_pred_train_inv.max()))
                    ax4.plot([lim_min, lim_max], [lim_min, lim_max], 'r--')
                    ax4.set_xlabel("Actual Values")
                    ax4.set_ylabel("Predicted Values")
                    ax4.set_title("Parity Plot (Training Data)")
                    st.pyplot(fig4)

                    # (5) Loss plot
                    fig5, ax5 = plt.subplots()
                    ax5.plot(history.history['loss'], label="Training Loss")
                    if 'val_loss' in history.history:
                        ax5.plot(history.history['val_loss'], label="Validation Loss")
                    ax5.set_title(f"Training vs Validation Loss ({loss_fn.upper()})")
                    ax5.set_xlabel("Epochs")
                    ax5.set_ylabel("Loss")
                    ax5.legend()
                    st.pyplot(fig5)

# ----------------------
# External Test Dataset
# ----------------------
st.subheader("📂 Upload External Test Data (Optional)")
test_file = st.file_uploader("Upload a separate CSV or Excel file for testing", type=['csv', 'xlsx', 'xls'])

if test_file is not None and "model" in st.session_state:
    df_test = load_any(test_file, purpose="External Test")
    st.write("📊 External Test Data Preview")
    st.dataframe(df_test.head())

    feat_cols = st.session_state["feature_cols"]
    target_col = st.session_state["target_col"]

    missing = [c for c in feat_cols + [target_col] if c not in df_test.columns]
    if missing:
        st.error(f"❌ Missing required columns: {missing}")
    else:
        df_test_num = df_test[feat_cols + [target_col]].copy()
        df_test_num = df_test_num.apply(pd.to_numeric, errors='coerce').dropna()

        if df_test_num.empty:
            st.error("❌ External test data has no valid numeric rows.")
        else:
            X_ext = df_test_num[feat_cols].values
            y_ext = df_test_num[[target_col]].values

            scaler_X = st.session_state["scaler_X"]
            scaler_y = st.session_state["scaler_y"]
            window_size = st.session_state["window_size"]
            model = st.session_state["model"]

            scaled_X_ext = scaler_X.transform(X_ext)
            scaled_y_ext = scaler_y.transform(y_ext)

            X_ext_win, y_ext_win = create_windows_multivariate(scaled_X_ext, scaled_y_ext, window_size)

            if X_ext_win.shape[0] == 0:
                st.error("❌ Window size too large for external test data.")
            else:
                n_features = len(feat_cols)
                X_ext_win = X_ext_win.reshape((X_ext_win.shape[0], X_ext_win.shape[1], n_features))
                y_pred_ext = model.predict(X_ext_win, verbose=0)

                y_ext_inv = scaler_y.inverse_transform(y_ext_win.reshape(-1, 1))
                y_pred_ext_inv = scaler_y.inverse_transform(y_pred_ext)

                r2_ext = r2_score(y_ext_inv, y_pred_ext_inv)
                st.success(f"📊 External Test R² = {r2_ext:.4f}")

                fig6, ax6 = plt.subplots()
                ax6.plot(y_ext_inv, label="Actual (External Test)", color="blue")
                ax6.plot(y_pred_ext_inv, label="Prediction", color="red")
                ax6.legend()
                ax6.set_title("Prediction vs Actual (External Test)")
                st.pyplot(fig6)
