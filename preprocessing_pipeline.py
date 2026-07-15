import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import zscore
import category_encoders as ce

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("dataset.csv")

print("Original Dataset:")
print(df)

# -----------------------------
# Memory Optimization
# -----------------------------
print("\nMemory Before:")
print(df.memory_usage(deep=True))

for col in df.select_dtypes(include=['int64']).columns:
    df[col] = pd.to_numeric(df[col], downcast='integer')

for col in df.select_dtypes(include=['float64']).columns:
    df[col] = pd.to_numeric(df[col], downcast='float')

print("\nMemory After:")
print(df.memory_usage(deep=True))

# -----------------------------
# Convert Object to Category
# -----------------------------
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype("category")

# -----------------------------
# Missing Value Imputation
# -----------------------------
num_cols = df.select_dtypes(include=['int', 'float']).columns
cat_cols = df.select_dtypes(include=['category']).columns

# Mean Imputation
mean_df = df.copy()
mean_df[num_cols] = SimpleImputer(strategy='mean').fit_transform(mean_df[num_cols])

# Median Imputation
median_df = df.copy()
median_df[num_cols] = SimpleImputer(strategy='median').fit_transform(median_df[num_cols])

# KNN Imputation
knn_df = df.copy()
knn_df[num_cols] = KNNImputer(n_neighbors=2).fit_transform(knn_df[num_cols])

# Fill categorical missing values
cat_imputer = SimpleImputer(strategy='most_frequent')
knn_df[cat_cols] = cat_imputer.fit_transform(knn_df[cat_cols])

print("\nAfter KNN Imputation:")
print(knn_df)

# -----------------------------
# Z-Score Outlier Detection
# -----------------------------
z = np.abs(zscore(knn_df[num_cols]))

z_df = knn_df[(z < 3).all(axis=1)]

print("\nDataset After Z-Score:")
print(z_df)

# -----------------------------
# IQR Outlier Detection
# -----------------------------
Q1 = knn_df[num_cols].quantile(0.25)
Q3 = knn_df[num_cols].quantile(0.75)

IQR = Q3 - Q1

iqr_df = knn_df[
    ~(
        (knn_df[num_cols] < (Q1 - 1.5 * IQR)) |
        (knn_df[num_cols] > (Q3 + 1.5 * IQR))
    ).any(axis=1)
]

print("\nDataset After IQR:")
print(iqr_df)

# -----------------------------
# One Hot Encoding
# -----------------------------
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

encoded = encoder.fit_transform(knn_df[cat_cols])

encoded_df = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out(cat_cols)
)

print("\nOne Hot Encoded Data:")
print(encoded_df)

# -----------------------------
# Target Encoding
# -----------------------------
if "Target" in knn_df.columns:

    encoder = ce.TargetEncoder(cols=cat_cols)

    target_encoded = encoder.fit_transform(
        knn_df.drop("Target", axis=1),
        knn_df["Target"]
    )

    print("\nTarget Encoded Data:")
    print(target_encoded)

print("\nPROJECT COMPLETED SUCCESSFULLY")