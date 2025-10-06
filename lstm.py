import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

st.title("📈 LSTM Time Series Demo (Multi-feature)")

# ----------------------
# Upload main dataset
# ----------------------
st.subheader("📂 Upload Training Data (CSV or Excel)")
main_file = st.file_uploader("Upload a CSV or Excel file for training", type=['csv', 'xlsx', 'xls'])

if main_file is not None:
    # Excel Sheet Selection
    if any(main_file.name.endswith(x) for x in [".xlsx", ".xls"]):
        excel_file = pd.ExcelFile(main_file)
        sheet_names = excel_file.sheet_names
        sheet_choice = st.selectbox("Select Sheet for Training Data:", sheet_names)
        df_series = excel_file.parse(sheet_choice)
    else:
        df_series = pd.read_csv(main_file)

    st.write("📊 Training Data Preview", df_series.head())

    # Multi-column selection
    feature_cols = st.multiselect(
        "Select Features (Input Columns):", 
        options=[col for col in df_series.columns],
        help="You can select more than one feature.",
    )
    target_col = st.selectbox(
        "Select Target (Output Column):", 
        options=[col for col in df_series.columns if col not in feature_cols],
    )

    if feature_cols and target_col:
        series_X = df_series[feature_cols]
        series_y = df_series[[target_col]]

        # Plot first feature
        fig, ax = plt.subplots()
        if len(series_X) > 2000:
            ax.plot(series_X.iloc[::10, 0], label=f"Sampled Feature: {feature_cols[0]}", color='blue')
        else:
            ax.plot(series_X.iloc[:, 0], label=f"Feature: {feature_cols[0]}", color='blue')
        ax.set_title(f"First Feature Time Series ({feature_cols[0]})")
        ax.legend()
        st.pyplot(fig)

        # ----------------------
        # Model Hyperparameters
        # ----------------------
        st.subheader("⚙️ LSTM Hyperparameters")
        window_size = st.slider("Window Size", min_value=5, max_value=50, value=20)
        num_neurons = st.slider("Number of LSTM Neurons", min_value=10, max_value=200, value=50, step=10)
        num_layers = st.slider("Number of LSTM Layers", min_value=1, max_value=3, value=1)
        activation_fn = st.selectbox("Activation Function", ["tanh", "relu", "sigmoid"])
        learning_rate = st.slider("Learning Rate", min_value=0.0001, max_value=0.01, value=0.001, step=0.0001, format="%.4f")
        batch_size = st.slider("Batch Size", min_value=8, max_value=128, value=16, step=8)
        num_epochs = st.slider("Epochs", min_value=5, max_value=100, value=10, step=5)
        loss_fn = st.selectbox("Loss Function", ["mse", "mae"])

        # Scale inputs and outputs
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        scaled_X = scaler_X.fit_transform(series_X)
        scaled_y = scaler_y.fit_transform(series_y)

        # Split train/test (80/20)
        train_size = int(len(scaled_X) * 0.8)
        train_X, test_X = scaled_X[:train_size], scaled_X[train_size:]
        train_y, test_y = scaled_y[:train_size], scaled_y[train_size:]

        # Function to build windows (Multivariate)
        def create_windows_multivariate(X, y, window):
            Xs, ys = [], []
            for i in range(len(X) - window):
                Xs.append(X[i:i+window, :])
                ys.append(y[i+window, 0])
            return np.array(Xs), np.array(ys)

        # Build train and test sets
        X_train, y_train = create_windows_multivariate(train_X, train_y, window_size)
        X_test, y_test = create_windows_multivariate(test_X, test_y, window_size)

        # ----------------------
        # Train model
        # ----------------------
        if st.button("Train LSTM Model"):
            if X_train.shape[0] == 0 or X_test.shape[0] == 0:
                st.error("❌ Window size is too large for the dataset. Reduce the slider value.")
            else:
                # Model input: (batch, timesteps, features)
                X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], len(feature_cols)))
                X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], len(feature_cols)))

                # Build the LSTM model
                model = Sequential()
                for i in range(num_layers):
                    return_sequences = (i < num_layers - 1)
                    model.add(
                        LSTM(num_neurons,
                             activation=activation_fn,
                             return_sequences=return_sequences,
                             input_shape=(window_size, len(feature_cols)) if i==0 else None)
                    )
                model.add(Dense(1))

                optimizer = Adam(learning_rate=learning_rate)
                model.compile(loss=loss_fn, optimizer=optimizer)

                # Train model
                history = model.fit(
                    X_train, y_train,
                    epochs=num_epochs,
                    batch_size=batch_size,
                    verbose=0,
                    validation_split=0.2,
                )

                st.session_state["model"] = model
                st.session_state["scaler_X"] = scaler_X
                st.session_state["scaler_y"] = scaler_y
                st.session_state["window_size"] = window_size
                st.session_state["feature_cols"] = feature_cols
                st.session_state["target_col"] = target_col

                # Evaluate
                y_pred_test = model.predict(X_test)
                y_pred_train = model.predict(X_train)

                # Inverse transform predictions
                y_pred_test_inv = scaler_y.inverse_transform(y_pred_test)
                y_pred_train_inv = scaler_y.inverse_transform(y_pred_train)
                y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1))
                y_train_inv = scaler_y.inverse_transform(y_train.reshape(-1, 1))

                r2_train = r2_score(y_train_inv, y_pred_train_inv)
                r2_test = r2_score(y_test_inv, y_pred_test_inv)
                st.success(f"✅ Model trained! Train R² = {r2_train:.4f}, Test R² = {r2_test:.4f}")

                # Plot test results
                fig1, ax1 = plt.subplots()
                ax1.plot(y_test_inv, label='Actual (Test)', color='orange')
                ax1.plot(y_pred_test_inv, label='LSTM Prediction', color='red')
                ax1.legend()
                ax1.set_title("Prediction vs Actual (Internal Test)")
                st.pyplot(fig1)

                # Parity plot (Test)
                fig2, ax2 = plt.subplots()
                ax2.scatter(y_test_inv, y_pred_test_inv, alpha=0.6, color='green')
                ax2.plot([y_test_inv.min(), y_test_inv.max()],
                         [y_test_inv.min(), y_test_inv.max()], 'r--')
                ax2.set_xlabel("Actual Values")
                ax2.set_ylabel("Predicted Values")
                ax2.set_title("Parity Plot (Internal Test)")
                st.pyplot(fig2)

                # Training vs validation loss
                fig3, ax3 = plt.subplots()
                ax3.plot(history.history['loss'], label="Training Loss")
                ax3.plot(history.history['val_loss'], label="Validation Loss")
                ax3.set_title(f"Training vs Validation Loss ({loss_fn.upper()})")
                ax3.set_xlabel("Epochs")
                ax3.set_ylabel("Loss")
                ax3.legend()
                st.pyplot(fig3)

                # Error distributions
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
            if any(test_file.name.endswith(x) for x in [".xlsx", ".xls"]):
                excel_file_test = pd.ExcelFile(test_file)
                sheet_names_test = excel_file_test.sheet_names
                sheet_choice_test = st.selectbox("Select Sheet for External Test Data:", sheet_names_test)
                df_test = excel_file_test.parse(sheet_choice_test)
            else:
                df_test = pd.read_csv(test_file)

            st.write("📊 External Test Data Preview", df_test.head())

            feat_cols = st.session_state["feature_cols"]
            target_col = st.session_state["target_col"]
            if any([col not in df_test.columns for col in feat_cols+[target_col]]):
                st.error(f"❌ Columns required: {feat_cols + [target_col]} not found in external test file.")
            else:
                X_ext = df_test[feat_cols]
                y_ext = df_test[[target_col]]
                scaler_X = st.session_state["scaler_X"]
                scaler_y = st.session_state["scaler_y"]
                window_size = st.session_state["window_size"]
                model = st.session_state["model"]

                # Scale
                scaled_X_ext = scaler_X.transform(X_ext)
                scaled_y_ext = scaler_y.transform(y_ext)

                # Windowing
                def create_windows_multivariate_eval(X, y, window):
                    Xs, ys = [], []
                    for i in range(len(X) - window):
                        Xs.append(X[i:i+window, :])
                        ys.append(y[i+window, 0])
                    return np.array(Xs), np.array(ys)

                X_ext_win, y_ext_win = create_windows_multivariate_eval(scaled_X_ext, scaled_y_ext, window_size)
                if X_ext_win.shape[0] == 0:
                    st.error("❌ Window size too large for external test file.")
                else:
                    X_ext_win = X_ext_win.reshape((X_ext_win.shape[0], X_ext_win.shape[1], len(feat_cols)))
                    # Predict
                    y_pred_ext = model.predict(X_ext_win)
                    y_ext_inv = scaler_y.inverse_transform(y_ext_win.reshape(-1, 1))
                    y_pred_ext_inv = scaler_y.inverse_transform(y_pred_ext)
                    r2_ext = r2_score(y_ext_inv, y_pred_ext_inv)
                    st.success(f"📊 External Test R² = {r2_ext:.4f}")

                    # Plot
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

