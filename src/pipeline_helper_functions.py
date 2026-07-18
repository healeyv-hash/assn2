import mlflow
import mlflow.sklearn
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import xgboost as xgb
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer

# Task 1: EDA Functions

# Function that shows what types of features there are and basic summary statistics
def eda_summary(df):
  print('---Feature types---')
  print(df.info())
  print('---Summary statistics---')
  print(df.describe().T)

# Function showing a count of missingness by feature
def eda_missingness(df):
  print('---Missing count by feature---')
  print(df.isnull().sum())

# Function showing a distribution (histogram) of numerical variables
def eda_numer_distribution(df):
  numeric_df = df.select_dtypes(include="number")

  for column in numeric_df.columns:
      plt.figure(figsize=(6, 4))
      plt.hist(numeric_df[column].dropna(), bins=50)
      plt.title(f"Distribution of {column}")
      plt.xlabel(column)
      plt.ylabel("Frequency")
      plt.grid(alpha=0.3)
      plt.show()

# Function showing a) the number of unique values for each categorical feature and b) examples of these values
def eda_categorical_exploration(df):
    categorical_df = df.select_dtypes(exclude="number")

    for column in categorical_df.columns:
        unique_vals = df[column].dropna().unique()
        examples = unique_vals[:5]
        print(f"{column} has {len(unique_vals)} unique values. Examples: {list(examples)}")



# Task 1 ct'd: data pre-processing functions
# Function that preprocesses data. At a high level, it drops missing values and irrelevant columns. It also removes outliers, cleans some of the string values in the data, and creates our target variable (total_lift)
def data_preprocessing(data):
    # Drop NA values for these key columns
    data = data.dropna(
        subset=[
            'region','age','weight','height','howlong',
            'gender','eat','background','experience',
            'schedule','deadlift','candj','snatch','backsq'
        ]
    )

    # drop columns where we are unsure of meaning or they are not relevant/informative
    data = data.drop(
        columns=[
            'affiliate','team','name',
            'fran','helen','grace','filthy50',
            'fgonebad'
        ]
    )

    # Remove outliers based on domain knowledge and data distributions
    data = data[data['weight'] < 1500]
    data = data[data['gender'] != '--']
    data = data[data['age'] >= 18]
    data = data[(data['height'] < 96) & (data['height'] > 48)]

    data = data[
        ((data['gender'] == 'Male') & (data['deadlift'] <= 1105)) |
        ((data['gender'] == 'Female') & (data['deadlift'] <= 636))
    ]

    data = data[(data['candj'] > 0) & (data['candj'] <= 395)]
    data = data[(data['snatch'] > 0) & (data['snatch'] <= 496)]
    data = data[(data['backsq'] > 0) & (data['backsq'] <= 1069)]
    data = data[(data['run400'] > 0) & (data['run400'] <= 600)]
    data = data[(data['run5k'] > 0) & (data['run400'] <= 4000)]
    data = data[(data['run5k'] >= 0) & (data['run400'] <= 150)]

    # Clean survey data by replacing "Decline to answer" with NA and then dropping NAs
    decline_dict = {'Decline to answer|': np.nan}
    data = data.replace(decline_dict)

    data = data.dropna(
        subset=[
            'background','experience',
            'schedule','howlong','eat'
        ]
    )

    # Build total_lift outcome variable
    data["total_lift"] = (
        data["deadlift"]
        + data["candj"]
        + data["snatch"]
        + data["backsq"]
    )

    return data

    # Several features, such as the "eat" feature, seem to contain multiple responses separated by "|"
# This function breaks up those responses and turns them into a series of dummy variables
# The resulting dummy var is equal to 1 if the respondent selected that option and 0 if not
# Note these sets of dummies will not be mutually exclusive, since respondents could select multiple features
def expand_multiselect(df, columns,sep="|"):
    df=df.copy()
    new_multiselect_cols = []
    for col in columns:
        categories = (
            df[col]
            .fillna("")
            .str.split(sep)
            .explode()
            .unique()
        )
        categories = [c for c in categories if c != ""]
        for cat in categories:
            new_col = f"{col}_{cat}"

            df[new_col] = (
                df[col]
                .fillna("")
                .str.contains(cat, regex=False)
                .astype(int)
            )
            new_multiselect_cols.append(new_col)
        df.drop(columns=col, inplace=True)

    return df


# Turn remaining categorical vars into dummies -- this will be helpful for our final XGBoost model
def create_dummies(df):
    categorical_features = df.select_dtypes(
        exclude="number"
    ).columns
    
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    
    # convert True/False to 1/0
    dummy_cols = df.columns.difference(df.select_dtypes(include="number").columns)
    df[dummy_cols] = df[dummy_cols].astype(int)
    
    return df


# Our final preprocessing step is to clean the column names, since some contain symbols that may cause issues down the line
def sanitize_column_names(df, max_len=60):
    df = df.copy()
    new_cols = {}
    for col in df.columns:
        clean = re.sub(r"[ ,;{}()\n\t=.]+", "_", col)
        clean = re.sub(r"_+", "_", clean).strip("_")
        new_cols[col] = clean[:max_len]
    df = df.rename(columns=new_cols)
    return df


# Task 3: Feature engineering functions
# Function that creates a basic form of the data set and identifies target feature
def build_features_v1(df):
    """v1: basic form of the data set: target feature (total_lift) added, categorical vars converted to dummies"""
    features = df.copy()
    features = features.drop(columns=["candj", "snatch", "backsq", "deadlift", "total_lift"])  # drop label (total_lift) and direct predictors of label
    return features

# Function that creates more advanced form of data set and identifies same target feature
def build_features_v2(df):
    """v2: advanced form of the data set: target feature added, categorical vars converted to dummies, new features such as height to weight ratio, speed to weight ratio, 5k to 400m speed ratio, and age bin created"""
    features = df.copy()
    features["height_weight_ratio"]=features["height"]/features["weight"]
    features["speed_weight_ratio"] = features["run5k"]/features["weight"]
    features['speed_ratio']=features["run5k"]/features["run400"]
    features["age_bin"] = pd.cut(df["age"], bins=[0, 25, 35, 45, 100], labels=["u25", "25-35", "35-45", "45+"])
    features = features.drop(columns=["age", "candj", "snatch", "backsq", "deadlift", "total_lift"])  # drop raw age, label (total_lift) and direct predictors of label
    features = create_dummies(features)  # one-hot encodes age_bin

    return features

# Task 3 ct'd: model building and evaluation functions

# This function finishes the preprocessing by turning remaining categorical variables into sets of dummies and scaling numeric vars. This will prevent feature dominance in the XGBoost model
def scale_data(X_train, X_test):

    # Numeric columns that actually need scaling (in other words, not the dummies we just created)
    numeric_features = [
        col for col in X_train.select_dtypes(include="number").columns
    ]

    # Transform with MinMaxScaler
    preprocessor = ColumnTransformer(
        transformers=[
            ("scale", MinMaxScaler(), numeric_features)
        ],
        verbose_feature_names_out=False
    )
    # Fit on train data and apply to test data to avoid data leakage
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed


# This function sets up the XGBoost model, fits it on the training data, and runs it on the test data
# It logs parameters and results to MLflow
# It also evaluates model performance using RMSE and R2
# NOTE: chunks of this code were borrowed from a class lab, such as tracking model performance and creating visuals
def run_xgb(X_train, X_test, y_train, y_test, feature_version, n_estimators=250, max_depth=5):
    # Establish the model: XGBoost Regressor    
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1
    )

    # Fit the model on training data
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Predict the model using the test data
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse) # Calculate RMSE by taking the square root of MSE
    r2 = r2_score(y_test, preds)

    with mlflow.start_run(run_name=f"xgb_datav{feature_version}_estimators_{n_estimators}_depth_{max_depth}"):
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("data feature version", feature_version)
        mlflow.log_param("model", "XGBoost")
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        mlflow.set_tag("experiment", "databricks-colab-test")

        # Plot prediction vs actual and log it
        plt.figure(figsize=(6, 4))
        plt.scatter(y_test, preds, alpha=0.3)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"Actual vs Predicted (xgb_datav{feature_version}_estimators_{n_estimators}_depth_{max_depth})")
        plt.grid(True)
        plot_path = f"prediction_plot_{feature_version}_{n_estimators}_{max_depth}.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)

    print("All experiments logged to Databricks MLflow.")

