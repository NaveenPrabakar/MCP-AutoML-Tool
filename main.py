#!/usr/bin/env python

import os
import json
import joblib
from typing import Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from mcp.server.fastmcp import FastMCP
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import GridSearchCV
import boto3
from pymongo import MongoClient
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import openai
import re

openai.api_key = "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"




S3_BUCKET = "nflfootballwebsite" 
s3_client = boto3.client("s3")


# MongoDB (local) config
mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["ml_modeler"]
datasets_collection = db["datasets"]



mcp = FastMCP("ML Modeler MCP")

@dataclass
class ModelConfig:
    model_type: str = "classification" 
    target_column: str = "target"

config = ModelConfig()
dataset_cache: Dict[str, pd.DataFrame] = {}
model_cache: Dict[str, Any] = {}

@mcp.tool(description="Upload a CSV dataset and cache it by name")
async def upload_dataset(name: str, csv_content: str) -> str:
    """
    Upload a dataset and cache it for future use.

    Parameters:
    - name: The dataset name
    - csv_content: The full CSV content as string
    """
    try:
        from io import StringIO
        df = pd.read_csv(StringIO(csv_content))
        dataset_cache[name] = df
        return f"Dataset '{name}' uploaded successfully with {df.shape[0]} rows and {df.shape[1]} columns."
    except Exception as e:
        return f"Error uploading dataset: {str(e)}"

@mcp.tool(description="Preview the first few rows of a dataset")
async def preview_dataset(name: str, rows: int = 5) -> str:
    """
    Preview a few rows of the dataset.

    Parameters:
    - name: The dataset name
    - rows: Number of rows to preview
    """
    df = dataset_cache.get(name)
    if df is None:
        return f"Dataset '{name}' not found."
    return df.head(rows).to_json(orient="records", indent=2)


@mcp.tool(description="Train a classification or regression model from dataset")
async def train_model(name: str, target_column: str, model_type: str = "classification", model_name: Optional[str] = None) -> str:
    """
    Train a model from the uploaded dataset.

    Parameters:
    - name: Dataset name
    - target_column: Name of the target column
    - model_type: 'classification' or 'regression'
    - model_name: Name of the specific model to use (optional)
    """
    df = dataset_cache.get(name)
    if df is None:
        return f"Dataset '{name}' not found."

    if target_column not in df.columns:
        return f"Target column '{target_column}' not in dataset."

    try:
        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_type == "classification":
            if model_name == "logistic_regression" or model_name is None:
                model = LogisticRegression(max_iter=1000)
            elif model_name == "random_forest":
                model = RandomForestClassifier(n_estimators=100)
            elif model_name == "svm":
                model = SVC(kernel='linear')
            elif model_name == "knn":
                model = KNeighborsClassifier(n_neighbors=5)
            elif model_name == "decision_tree":
                model = DecisionTreeClassifier(random_state=42)
            else:
                return f"Invalid classification model name: {model_name}"

            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            model_cache[name] = model
            return f"Classification model trained using {model_name}. Accuracy: {acc:.4f}"

        elif model_type == "regression":
            if model_name == "linear_regression" or model_name is None:
                model = LinearRegression()
            elif model_name == "random_forest":
                model = RandomForestRegressor(n_estimators=100)
            elif model_name == "svm":
                model = SVR(kernel='linear')
            elif model_name == "decision_tree":
                model = DecisionTreeRegressor(random_state=42)
            else:
                return f"Invalid regression model name: {model_name}"

            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            model_cache[name] = model
            return f"Regression model trained using {model_name}. MSE: {mse:.4f}"

        else:
            return f"Invalid model type: {model_type}. Use 'classification' or 'regression'."

    except Exception as e:
        return f"Error training model: {str(e)}"



@mcp.tool(description="Download a dataset from Kaggle and use it like a normal uploaded dataset")
async def download_kaggle_dataset(name: str, kaggle_url: str) -> str:
    """
    Downloads a dataset from Kaggle and loads the first CSV file into the dataset cache.

    Parameters:
    - name: The name to cache the dataset under
    - kaggle_url: The Kaggle dataset URL (e.g., https://www.kaggle.com/datasets/username/datasetname)
    """
    try:
        import os
        import pandas as pd
        import re
        import tempfile
        from kaggle.api.kaggle_api_extended import KaggleApi

        # Define a global or external dataset cache if not already defined
        global dataset_cache
        if 'dataset_cache' not in globals():
            dataset_cache = {}

        # Extract the dataset identifier from the URL
        match = re.search(r"kaggle\.com/datasets/([^/]+/[^/?#]+)", kaggle_url)
        if not match:
            return "Invalid Kaggle dataset URL. Expected format: https://www.kaggle.com/datasets/username/datasetname"
        dataset_identifier = match.group(1)

        # Authenticate with the Kaggle API
        api = KaggleApi()
        api.authenticate()

        # Download and extract dataset
        with tempfile.TemporaryDirectory() as tmp_dir:
            api.dataset_download_files(dataset_identifier, path=tmp_dir, unzip=True)

            # Look for a CSV file
            csv_files = [f for f in os.listdir(tmp_dir) if f.endswith(".csv")]
            if not csv_files:
                return "No CSV file found in the Kaggle dataset."

            csv_path = os.path.join(tmp_dir, csv_files[0])
            df = pd.read_csv(csv_path)

            # Store in dataset cache
            dataset_cache[name] = df

            return f"Kaggle dataset '{name}' loaded with {df.shape[0]} rows and {df.shape[1]} columns."

    except Exception as e:
        return f"Error downloading or processing Kaggle dataset: {str(e)}"
    

@mcp.tool(description="Clean a dataset by imputing missing values, removing duplicates, and encoding categoricals")
async def clean_dataset(name: str, encode_categoricals: bool = True) -> str:
    """
    Clean the dataset:
    - Impute missing values (mean for numeric, mode for categorical)
    - Remove duplicate rows
    - Encode categorical variables (optional)

    Parameters:
    - name: Name of the dataset
    - encode_categoricals: Whether to one-hot encode categorical columns (default: True)
    """
    df = dataset_cache.get(name)
    if df is None:
        return f"Dataset '{name}' not found."

    try:
        # Impute missing values
        for column in df.columns:
            if df[column].isnull().any():
                if df[column].dtype in ['float64', 'int64']:
                    df[column] = df[column].fillna(df[column].mean())
                elif df[column].dtype == 'object':  # Handle categorical features
                    df[column] = df[column].fillna(df[column].mode().iloc[0])

        # Remove duplicates
        df = df.drop_duplicates()

        # Encode categoricals
        if encode_categoricals:
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                df = pd.get_dummies(df, drop_first=True)  # Drop the first column to avoid multicollinearity


        # Update the dataset cache
        dataset_cache[name] = df

        return f"Dataset '{name}' cleaned with missing values imputed and duplicates removed. Final shape: {df.shape[0]} rows, {df.shape[1]} columns."

    except Exception as e:
        return f"Error cleaning dataset: {str(e)}"


@mcp.tool(description="Visualize the data distribution (histogram for numeric columns)")
async def visualize_data_distribution(name: str) -> str:
    """
    Generate histograms for each numeric column in the dataset.
    """
    df = dataset_cache.get(name)
    if df is None:
        return f"Dataset '{name}' not found."

    try:
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if not numeric_cols.any():
            return f"No numeric columns found in dataset '{name}'."
        
        plt.figure(figsize=(10, 6))
        for idx, col in enumerate(numeric_cols):
            plt.subplot(2, len(numeric_cols)//2, idx+1)
            sns.histplot(df[col], kde=True)
            plt.title(f"Distribution of {col}")
        
        plt.tight_layout()
        plt.show()
        return f"Data distribution visualization completed."
    
    except Exception as e:
        return f"Error during visualization: {str(e)}"
    

@mcp.tool(description="Generate a confusion matrix for the classification model")
async def plot_confusion_matrix(name: str) -> str:
    """
    Plot the confusion matrix for the trained classification model.
    """
    model = model_cache.get(name)
    df = dataset_cache.get(name)
    if model is None or df is None:
        return f"Model or dataset '{name}' not found."

    target = config.target_column
    if target not in df.columns:
        return f"Target column '{target}' not in dataset."

    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    try:
        preds = model.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=model.classes_, yticklabels=model.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

        return f"Confusion matrix visualized."

    except Exception as e:
        return f"Error during confusion matrix generation: {str(e)}"


@mcp.tool(description="Generate a summary of the dataset")
async def dataset_summary(name: str) -> str:
    """
    Generate a summary of the dataset, including basic statistics.
    """
    df = dataset_cache.get(name)
    if df is None:
        return f"Dataset '{name}' not found."

    try:
        summary = df.describe(include='all')  # Include both numeric and categorical columns
        return summary.to_json(orient='split', indent=2)
    
    except Exception as e:
        return f"Error generating dataset summary: {str(e)}"



@mcp.tool(description="Hyperparameter tuning with GridSearchCV")
async def hyperparameter_tuning(name: str, target: str, model_type: str = "classification", model_name: Optional[str] = None) -> str:
    """
    Perform hyperparameter tuning using GridSearchCV.
    """
    df = dataset_cache.get(name)
    if df is None:
        return f"Dataset '{name}' not found."

    if target not in df.columns:
        return f"Target column '{target}' not in dataset."

    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    try:
        if model_type == "classification":
            if model_name == "logistic_regression" or model_name is None:
                model = LogisticRegression(max_iter=1000)
                param_grid = {'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear', 'lbfgs'], 'penalty': ['l2', 'none']}
            elif model_name == "random_forest":
                model = RandomForestClassifier(n_estimators=100)
                param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
            elif model_name == "svm":
                model = SVC(kernel='linear')
                param_grid = {'C': [0.01, 0.1, 1, 10], 'kernel': ['linear', 'rbf']}
            elif model_name == "knn":
                model = KNeighborsClassifier(n_neighbors=5)
                param_grid = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
            elif model_name == "decision_tree":
                model = DecisionTreeClassifier(random_state=42)
                param_grid = {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
            else:
                return f"Invalid classification model name: {model_name}"

        elif model_type == "regression":
            if model_name == "linear_regression" or model_name is None:
                model = LinearRegression()
                param_grid = {'fit_intercept': [True, False], 'normalize': [True, False]}
            elif model_name == "random_forest":
                model = RandomForestRegressor(n_estimators=100)
                param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
            elif model_name == "svm":
                model = SVR(kernel='linear')
                param_grid = {'C': [0.01, 0.1, 1, 10], 'kernel': ['linear', 'rbf']}
            elif model_name == "decision_tree":
                model = DecisionTreeRegressor(random_state=42)
                param_grid = {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
            else:
                return f"Invalid regression model name: {model_name}"

        else:
            return f"Invalid model type. Choose 'classification' or 'regression'."

        grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_

        return f"Best hyperparameters for {model_name}: {best_params}"

    except Exception as e:
        return f"Error during hyperparameter tuning: {str(e)}"



@mcp.tool(description="Save a trained model to AWS S3 bucket")
async def save_model_to_s3(name: str) -> str:
    """
    Uploads a trained model to an AWS S3 bucket.
    """
    model = model_cache.get(name)
    if model is None:
        return f"No model found for dataset '{name}'."

    local_file = f"{name}_model.pkl"
    try:
        joblib.dump(model, local_file)
        s3_client.upload_file(local_file, S3_BUCKET, local_file)
        os.remove(local_file)
        return f"Model '{name}' uploaded to S3 bucket '{S3_BUCKET}' as '{local_file}'."
    except Exception as e:
        return f"Failed to upload model to S3: {str(e)}"


@mcp.tool(description="Save a dataset to MongoDB")
async def save_dataset_to_mongo(name: str) -> str:
    """
    Saves the full dataset to MongoDB as individual documents.
    """
    df = dataset_cache.get(name)
    if df is None:
        return f"Dataset '{name}' not found."

    try:
        # Convert DataFrame to dictionary records
        records = df.to_dict(orient='records')

        # Use a dedicated collection for each dataset, or store all in one with a dataset_name field
        collection_name = f"dataset_{name}"
        mongo_client.drop_database(collection_name)  # Optional: clear previous version
        collection = mongo_client[collection_name]["data"]

        # Insert records
        collection.insert_many(records)

        return f"Dataset '{name}' with {len(records)} records saved to MongoDB collection '{collection_name}.data'."
    except Exception as e:
        return f"Error saving dataset to MongoDB: {str(e)}"
    
@mcp.tool(description="Transform the dataset by applying one-hot encoding and scaling numerical features, don't transform the target variable")
async def transform_dataset(name: str, target: str, encode_categoricals: bool = True, normalize_numerics: bool = True) -> str:
    df = dataset_cache.get(name)
    if df is None:
        return f"Dataset '{name}' not found."

    try:
        # Assume the label column is the last column
        label_col = target

        # One-hot encoding for categorical columns (excluding label)
        if encode_categoricals:
            categorical_cols = df.select_dtypes(include=['object']).columns.difference([label_col])
            if len(categorical_cols) > 0:
                label_encoder = LabelEncoder()
                for col in categorical_cols:
                    df[col] = label_encoder.fit_transform(df[col])

        # Normalize numerical columns (excluding label)
        if normalize_numerics:
            numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.difference([label_col])
            if len(numerical_cols) > 0:
                scaler = StandardScaler()
                df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

        dataset_cache[name] = df

        return f"Dataset '{name}' transformed. Final shape: {df.shape[0]} rows, {df.shape[1]} columns."

    except Exception as e:
        return f"Error during dataset transformation: {str(e)}"

@mcp.tool(description="Perform automatic feature engineering using OpenAI API")
async def feature_engineering_with_openai(name: str) -> str:
    """
    Sends dataset information to OpenAI, receives a feature engineering script, executes it, and updates the dataset.
    """
    df = dataset_cache.get(name)
    if df is None:
        return f"Dataset '{name}' not found."

    try:
        # Build dataset summary to send
        summary_info = {
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "sample_rows": df.head(3).to_dict(orient="records")
        }

        # Create prompt
        prompt = f"""
You are a data scientist. Given the following dataset structure, return a Python function named `engineer_features(df)` that adds relevant feature-engineered columns to the DataFrame.

Respond ONLY with the function definition (no text or comments). It should take `df` as input and return `df`.

Dataset info:
{json.dumps(summary_info, indent=2)}
        """

        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        code = response["choices"][0]["message"]["content"]
        code = re.sub(r"^```(python)?\s*|```$", "", code.strip(), flags=re.MULTILINE)

        # Execute returned function
        local_vars = {}
        exec(code, {}, local_vars)
        if "engineer_features" not in local_vars:
            return "OpenAI did not return a valid function."

        # Apply transformation
        df = local_vars["engineer_features"](df)
        dataset_cache[name] = df

        return f"Feature engineering applied. Dataset '{name}' now has shape {df.shape}."

    except Exception as e:
        return f"Error during feature engineering: {str(e)}"



if __name__ == "__main__":
    print("Starting ML Modeler MCP Server...")
    mcp.run()


