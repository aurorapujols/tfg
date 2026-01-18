import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, RobustScaler

def process_numerical_features(X_train):
    numerical_variables = ["time", "width", "height", "mean_brightness", "bmin", "bmax"]

    num_features_df = X_train[["filename"] + numerical_variables + ["class"]]
    num_features_df = num_features_df.sort_values(by="filename").reset_index(drop=True)

    # To make the numerical features smooth, reduce skew and heavy tails, make them robust to outliers, and put them on comparable scale
    num_preprocess = Pipeline([
        # To make each distribution more Gaussian-like
        #  - reduces skew, stabilizes variance, makes features more symmetric and easier for linear layers to learn from
        ("power", PowerTransformer(method="yeo-johnson")),

        # Scales each feature using median and interquartile range (instead of mean and std)
        #   - resistant to outliers (many in meteor data), so it prevents them from dominating scale  
        ("scale", RobustScaler())
    ])
    
    X_num = num_features_df[numerical_variables]
    X_num_norm = num_preprocess.fit_transform(X_num)
    X_train_norm = pd.DataFrame(X_num_norm, columns=numerical_variables)
    X_train_norm["class"] = num_features_df["class"].values
    X_train_norm["filename"] = num_features_df["filename"].values

    return X_train_norm

# dataset = pd.read_csv("../../../data/upftfg26/apujols/processed/dataset_temp.csv", sep=";")
# print(f"Dataset rows: {len(dataset)}")
# X_norm.to_csv("../../../data/upftfg26/apujols/processed/num_features.csv", sep=";", index=False)