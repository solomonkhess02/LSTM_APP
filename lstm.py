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
                st.error(f"❌ Missing required columns in external test file: {missing}")
            else:
                df_test_num = df_test[feat_cols + [target_col]].copy()
                df_test_num = df_test_num.apply(pd.to_numeric, errors='coerce').dropna()
                if df_test_num.empty:
                    st.error("❌ External test data has no valid numeric rows after cleaning.")
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
                        st.error("❌ Window size too large for external test file length.")
                    else:
                        n_features = len(feat_cols)
                        X_ext_win = X_ext_win.reshape((X_ext_win.shape[0], X_ext_win.shape[1], n_features))

                        y_pred_ext = model.predict(X_ext_win, verbose=0)
                        y_ext_inv = scaler_y.inverse_transform(y_ext_win.reshape(-1, 1))
                        y_pred_ext_inv = scaler_y.inverse_transform(y_pred_ext)
                        r2_ext = r2_score(y_ext_inv, y_pred_ext_inv)
                        st.success(f"📊 External Test R² = {r2_ext:.4f}")

                        fig5, ax5 = plt.subplots()
                        ax5.plot(y_ext_inv, label="Actual (External Test)", color="blue")
                        ax5.plot(y_pred_ext_inv, label="Prediction", color="red")
                        ax5.legend()
                        ax5.set_title("Prediction vs Actual (External Test)")
                        st.pyplot(fig5)

else:
    st.info("👆 Please upload a CSV or Excel file to start.")


# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from tensorflow.keras.optimizers import Adam
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import r2_score

# st.title("📈 LSTM Time Series Demo")

# # ----------------------
# # Upload main dataset
# # ----------------------
# st.subheader("📂 Upload Training Data (CSV or Excel)")
# main_file = st.file_uploader("Upload a CSV or Excel file for training", type=['csv', 'xlsx', 'xls'])

# if main_file is not None:
#     # Load training dataset
#     if main_file.name.endswith(".csv"):
#         df_series = pd.read_csv(main_file)
#     else:
#         df_series = pd.read_excel(main_file)

#     st.write("📊 Training Data Preview", df_series.head())

#     # Let user pick the column
#     col_choice = st.selectbox("Select the column to use for training:", df_series.columns)

#     # Use the selected column
#     series = df_series[[col_choice]]

#     # Plot input series (downsample if large)
#     fig, ax = plt.subplots()
#     if len(series) > 2000:
#         ax.plot(series[col_choice].iloc[::10], label='Input Series (sampled)', color='blue')
#     else:
#         ax.plot(series[col_choice], label='Input Series', color='blue')
#     ax.set_title(f"Raw Input Time Series ({col_choice})")
#     ax.legend()
#     st.pyplot(fig)

#     # ----------------------
#     # Model Hyperparameters
#     # ----------------------
#     st.subheader("⚙️ LSTM Hyperparameters")

#     window_size = st.slider("Window Size", min_value=5, max_value=50, value=20)
#     num_neurons = st.slider("Number of LSTM Neurons", min_value=10, max_value=200, value=50, step=10)
#     num_layers = st.slider("Number of LSTM Layers", min_value=1, max_value=3, value=1)
#     activation_fn = st.selectbox("Activation Function", ["tanh", "relu", "sigmoid"])
#     learning_rate = st.slider("Learning Rate", min_value=0.0001, max_value=0.01,
#                               value=0.001, step=0.0001, format="%.4f")
#     batch_size = st.slider("Batch Size", min_value=8, max_value=128, value=16, step=8)
#     num_epochs = st.slider("Epochs", min_value=5, max_value=100, value=10, step=5)
#     loss_fn = st.selectbox("Loss Function", ["mse", "mae"])

#     # Scale the full series
#     scaler = MinMaxScaler()
#     scaled_series = scaler.fit_transform(series.values.reshape(-1, 1))

#     # Split train/test (80/20)
#     train_size = int(len(scaled_series) * 0.8)
#     train, test = scaled_series[:train_size], scaled_series[train_size:]

#     # Function to build windows
#     def create_windows(series, window):
#         X, y = [], []
#         for i in range(len(series) - window):
#             X.append(series[i:i+window])
#             y.append(series[i+window])
#         return np.array(X), np.array(y)

#     # Build train and test sets
#     X_train, y_train = create_windows(train, window_size)
#     X_test, y_test = create_windows(test, window_size)

#     # ----------------------
#     # Train model
#     # ----------------------
#     if st.button("Train LSTM Model"):
#         if X_train.shape[0] == 0 or X_test.shape[0] == 0:
#             st.error("❌ Window size is too large for the dataset. Reduce the slider value.")
#         else:
#             # Reshape for LSTM
#             X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
#             X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

#             # Build LSTM model dynamically
#             model = Sequential()
#             for i in range(num_layers):
#                 return_sequences = (i < num_layers - 1)
#                 model.add(LSTM(num_neurons,
#                                activation=activation_fn,
#                                return_sequences=return_sequences,
#                                input_shape=(window_size, 1) if i == 0 else None))
#             model.add(Dense(1))

#             optimizer = Adam(learning_rate=learning_rate)
#             model.compile(loss=loss_fn, optimizer=optimizer)

#             # Train model
#             history = model.fit(X_train, y_train,
#                                 epochs=num_epochs,
#                                 batch_size=batch_size,
#                                 verbose=0,
#                                 validation_split=0.2)

#             # Save model + scaler to session state ✅
#             st.session_state["model"] = model
#             st.session_state["scaler"] = scaler
#             st.session_state["window_size"] = window_size
#             st.session_state["col_choice"] = col_choice

#             # Predict on internal test set
#             y_pred_train = model.predict(X_train)
#             y_pred_test = model.predict(X_test)

#             # Inverse scaling
#             y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
#             y_pred_train_inv = scaler.inverse_transform(y_pred_train)
#             y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
#             y_pred_test_inv = scaler.inverse_transform(y_pred_test)

#             # Metrics
#             r2_train = r2_score(y_train_inv, y_pred_train_inv)
#             r2_test = r2_score(y_test_inv, y_pred_test_inv)
#             st.success(f"✅ Model trained! Train R² = {r2_train:.4f}, Test R² = {r2_test:.4f}")

#             # Plot results (Test)
#             fig1, ax1 = plt.subplots()
#             ax1.plot(y_test_inv, label='Actual (Test)', color='orange')
#             ax1.plot(y_pred_test_inv, label='LSTM Prediction', color='red')
#             ax1.legend()
#             ax1.set_title("Prediction vs Actual (Internal Test)")
#             st.pyplot(fig1)

#             # Parity plot (Test)
#             fig2, ax2 = plt.subplots()
#             ax2.scatter(y_test_inv, y_pred_test_inv, alpha=0.6, color='green')
#             ax2.plot([y_test_inv.min(), y_test_inv.max()],
#                      [y_test_inv.min(), y_test_inv.max()], 'r--')
#             ax2.set_xlabel("Actual Values")
#             ax2.set_ylabel("Predicted Values")
#             ax2.set_title("Parity Plot (Internal Test)")
#             st.pyplot(fig2)

#             # Training vs validation loss
#             fig3, ax3 = plt.subplots()
#             ax3.plot(history.history['loss'], label="Training Loss")
#             ax3.plot(history.history['val_loss'], label="Validation Loss")
#             ax3.set_title(f"Training vs Validation Loss ({loss_fn.upper()})")
#             ax3.set_xlabel("Epochs")
#             ax3.set_ylabel("Loss")
#             ax3.legend()
#             st.pyplot(fig3)

#             # Error distributions
#             train_errors = y_train_inv.flatten() - y_pred_train_inv.flatten()
#             test_errors = y_test_inv.flatten() - y_pred_test_inv.flatten()

#             fig4, ax4 = plt.subplots()
#             ax4.hist(train_errors, bins=30, alpha=0.7, label="Train Errors", color="blue")
#             ax4.hist(test_errors, bins=30, alpha=0.7, label="Test Errors", color="red")
#             ax4.set_title("Error Distribution (Train vs Test)")
#             ax4.set_xlabel("Error")
#             ax4.set_ylabel("Frequency")
#             ax4.legend()
#             st.pyplot(fig4)

#     # ----------------------
#     # External Test Dataset
#     # ----------------------
#     st.subheader("📂 Upload External Test Data (Optional)")
#     test_file = st.file_uploader("Upload a separate CSV or Excel file for testing", type=['csv', 'xlsx', 'xls'])

#     if test_file is not None and "model" in st.session_state:
#         if test_file.name.endswith(".csv"):
#             df_test = pd.read_csv(test_file)
#         else:
#             df_test = pd.read_excel(test_file)

#         st.write("📊 External Test Data Preview", df_test.head())

#         # Ensure same column is present
#         if st.session_state["col_choice"] not in df_test.columns:
#             st.error(f"❌ Column '{st.session_state['col_choice']}' not found in external test file.")
#         else:
#             series_test = df_test[[st.session_state["col_choice"]]]
#             scaler = st.session_state["scaler"]
#             window_size = st.session_state["window_size"]
#             model = st.session_state["model"]

#             # Scale
#             scaled_test = scaler.transform(series_test.values.reshape(-1, 1))

#             # Create windows
#             X_ext, y_ext = create_windows(scaled_test, window_size)
#             if X_ext.shape[0] == 0:
#                 st.error("❌ Window size too large for external test file.")
#             else:
#                 X_ext = X_ext.reshape((X_ext.shape[0], X_ext.shape[1], 1))

#                 # Predict
#                 y_pred_ext = model.predict(X_ext)
#                 y_ext_inv = scaler.inverse_transform(y_ext.reshape(-1, 1))
#                 y_pred_ext_inv = scaler.inverse_transform(y_pred_ext)

#                 # R² score
#                 r2_ext = r2_score(y_ext_inv, y_pred_ext_inv)
#                 st.success(f"📊 External Test R² = {r2_ext:.4f}")

#                 # Plot
#                 fig5, ax5 = plt.subplots()
#                 ax5.plot(y_ext_inv, label="Actual (External Test)", color="blue")
#                 ax5.plot(y_pred_ext_inv, label="Prediction", color="red")
#                 ax5.legend()
#                 ax5.set_title("Prediction vs Actual (External Test)")
#                 st.pyplot(fig5)

# else:
#     st.info("👆 Please upload a CSV or Excel file to start.")


