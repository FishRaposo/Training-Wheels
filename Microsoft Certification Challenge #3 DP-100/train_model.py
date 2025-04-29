import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import os

# Set MLflow experiment
mlflow.set_experiment("IceCreamSalesPrediction")

# Load dataset using absolute path
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "Inputs", "ice_cream_sales.csv")
data = pd.read_csv(data_path)

# Prepare features (X) and target (y)
X = data[["temperature"]]
y = data["sales"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])

# Start MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Log metrics
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    
    # Create an input example
    input_example = X_train.iloc[:1]
    
    # Log model with signature and input example
    mlflow.sklearn.log_model(
        pipeline, 
        "ice_cream_sales_model",
        input_example=input_example,
        signature=mlflow.models.infer_signature(X_train, y_train)
    )
    
    print(f"Model trained. MSE: {mse:.2f}, R2: {r2:.2f}")