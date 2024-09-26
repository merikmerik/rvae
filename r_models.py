# Define regression models

# Random Forest Regressor: An ensemble model using multiple decision trees, Tested with two configurations
# MLP Regressor: A neural network model.Tested with two configurations: MLP1 and MLP2

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_processor import load_data

class ModelTrainer:
    def __init__(self, model, X_train, y_train):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X):
        return self.model.predict(X)

class MetricsEvaluator:
    def __init__(self, y_true):
        self.y_true = y_true

    def evaluate(self, predictions):
        mae = mean_absolute_error(self.y_true, predictions)
        mse = mean_squared_error(self.y_true, predictions)
        r2 = r2_score(self.y_true, predictions)
        return mae, mse, r2

class MLPModelTrainer:
    def __init__(self, parameters, X_train, X_test, y_train, y_test):
        self.parameters = parameters
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.metrics = []
    
    def train_and_evaluate(self):
        for param in self.parameters:
            regr = MLPRegressor(
                hidden_layer_sizes=param['hidden_layer_sizes'],
                learning_rate_init=param['learning_rate_init'],
                early_stopping=True,
                random_state=1,
                max_iter=1500
            )

            print(f"Training model: {param['model']}")
            regr.fit(self.X_train, self.y_train)

            pred_train = regr.predict(self.X_train)
            pred_test = regr.predict(self.X_test)

            evaluator_train = MetricsEvaluator(self.y_train)
            evaluator_test = MetricsEvaluator(self.y_test)

            mae_train, mse_train, r2_train = evaluator_train.evaluate(pred_train)
            mae_test, mse_test, r2_test = evaluator_test.evaluate(pred_test)

            print(f"Metrics for {param['model']} - Train: MAE={mae_train}, MSE={mse_train}, R2={r2_train}")
            print(f"Metrics for {param['model']} - Test: MAE={mae_test}, MSE={mse_test}, R2={r2_test}")

            self.metrics.append({
                'Model': param['model'],
                'Hidden Layers': param['hidden_layer_sizes'],
                'Initial Learning Rate': param['learning_rate_init'],
                'MAE (Train)': round(mae_train, 2),
                'MSE (Train)': round(mse_train, 2),
                'R-squared (Train)': round(r2_train, 2),
                'MAE (Test)': round(mae_test, 2),
                'MSE (Test)': round(mse_test, 2),
                'R-squared (Test)': round(r2_test, 2)
            })

    def display_metrics(self):
        mlp_metrics_df = pd.DataFrame(self.metrics)
        print("\n", mlp_metrics_df.to_string(index=False), "\n")

def preprocess(df, feature_cols, target_col):
    X = df.iloc[:, feature_cols].values
    y = df.iloc[:, target_col].values.ravel() 

    # Remove outliers
    # X, y = remove_outliers(X, y)

    # Scale features and target
        from sklearn.preprocessing import StandardScaler
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()  # Reshape and ravel

    return train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

if __name__ == "__main__":
    # Load and preprocess the data
    dataframe_4 = load_data()  # Use load_data from data_processor.py
    
    # Debugging output
    print("Data loaded successfully")
    print(dataframe_4.head())  # Print the first few rows of the dataframe

    feature_columns = [2, 3]  # GRL and RES columns
    target_column = [1]       # RHOB column

    X_train, X_test, y_train, y_test = preprocess(dataframe_4, feature_columns, target_column)

    # Debugging output
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Model Training and Prediction
    rf_model_1 = RandomForestRegressor(n_estimators=20, random_state=42)
    rf_model_2 = RandomForestRegressor(n_estimators=120, random_state=42)

    trainer_1 = ModelTrainer(rf_model_1, X_train, y_train)
    trainer_1.train_model()
    rf_pred_train = trainer_1.predict(X_train)
    rf_pred_test = trainer_1.predict(X_test)

    evaluator = MetricsEvaluator(y_test)
    mae, mse, r2 = evaluator.evaluate(rf_pred_test)
    print(f"RandomForestRegressor - MAE: {mae}, MSE: {mse}, R-squared: {r2}")

    # Define parameters for MLP
    parameters = [
        {'model': 'MLP1', 'hidden_layer_sizes': (64, 32), 'learning_rate_init': 0.0001},
        {'model': 'MLP2', 'hidden_layer_sizes': (64, 32, 16), 'learning_rate_init': 0.001}
    ]
    mlp_trainer = MLPModelTrainer(parameters, X_train, X_test, y_train, y_test)
    mlp_trainer.train_and_evaluate()
    mlp_trainer.display_metrics()
