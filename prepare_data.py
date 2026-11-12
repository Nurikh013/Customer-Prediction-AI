
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler





def load_training_data():
    data_path = Path(__file__).parent / "data" / "bank-full.csv"

    df = pd.read_csv(data_path, sep=";")

    X = df.drop(columns=["y", "duration"])

    y = df["y"].map({
        "no": 0,
        "yes": 1
    })

    return X, y


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test



def build_preprocessor(X_train):
    categorical_columns = X_train.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    numerical_columns = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    print("\nCategorical columns:")
    print(categorical_columns)

    print("\nNumerical columns:")
    print(numerical_columns)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns
            ),
            (
                "numerical",
                StandardScaler(),
                numerical_columns
            )
        ]
    )

    return preprocessor





if __name__ == "__main__":
    X, y = load_training_data()

    print("Features shape:")
    print(X.shape)

    print("\nTarget shape:")
    print(y.shape)

    print("\nFeature columns:")
    print(X.columns.tolist())

    print("\nTarget distribution:")
    print(y.value_counts())


    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\nTraining features shape:")
    print(X_train.shape)

    print("\nTesting features shape:")
    print(X_test.shape)

    print("\nTraining target shape:")
    print(y_train.shape)

    print("\nTesting target shape:")
    print(y_test.shape)

    print("\nTraining subscription rate (%):")
    print(round(y_train.mean() * 100, 2))

    print("\nTesting subscription rate (%):")
    print(round(y_test.mean() * 100, 2))


    preprocessor = build_preprocessor(X_train)

    X_train_encoded = preprocessor.fit_transform(X_train)

    X_test_encoded = preprocessor.transform(X_test)

    print("\nOriginal training shape:")
    print(X_train.shape)

    print("\nEncoded training shape:")
    print(X_train_encoded.shape)

    print("\nEncoded testing shape:")
    print(X_test_encoded.shape)


