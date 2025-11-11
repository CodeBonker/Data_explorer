import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

class DataPreprocessor:
    """
    Automatically cleans and prepares data for ML models
    """

    def __init__(self, df: pd.DataFrame, target: str):
        self.df = df.copy()
        self.target = target
        self.report = {}
        self.text_features = []
        self.categorical_features = []
        self.numeric_features = []
        self.datetime_features = []

    def inspect_column_types(self):
        """Detect inconsistent or mixed-type columns and log conversion actions"""
        type_summary = {}
        conversion_actions = []

        for col in self.df.columns:
            if col == self.target:
                continue
            
            #  detect mixed types
            unique_types = self.df[col].dropna().map(type).value_counts()
            if len(unique_types) > 1:
                type_summary[col] =[str(t) for t in unique_types.index]

                conversion_actions.append(
                    f"Column '{col}' has mixed types: {list(unique_types.index)} — converted to string for uniformity")
                
                self.df[col] = self.df[col].astype(str)

            #  Detect object columns that look numeric
            elif self.df[col].dtype == object:
                try:
                    self.df[col] = pd.to_numeric(self.df[col])
                    conversion_actions.append(f"Column '{col}' converted from object to numeric.")
                except Exception:
                    pass

        if conversion_actions:
            self.report["dtype_corrections"] = conversion_actions

        else:
            self.report["dtype_corrections"] = ["No dtype corrections were needed."]



    def detect_column_types(self):
        for col in self.df.columns:
            if col == self.target:
                continue

            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.numeric_features.append(col)

            elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                self.datetime_features.append(col)

            elif self.df[col].dtype == object:
                avg_len = self.df[col].dropna().astype(str).str.len().mean()
                if avg_len > 25:
                    self.text_features.append(col)
                else:
                    self.categorical_features.append(col)
            

            self.report["column_summary"] = {
                "numeric": len(self.numeric_features),
                "categorical": len(self.categorical_features),
                "text": len(self.text_features),
                "datetime": len(self.datetime_features),
            }


    def handle_missing(self):

        for col in self.numeric_features:
            self.df[col] = self.df[col].fillna(self.df[col].median())

        for col in self.categorical_features:
            self.df[col] = self.df[col].fillna("missing")

        for col in self.text_features:
            self.df[col] = self.df[col].fillna("")

        self.report["missing_handled"] = True



    def encode_categoricals(self):
        encoded_df = pd.DataFrame(index=self.df.index)

        for col in self.categorical_features:
            self.df[col] = self.df[col].astype(str)  # ensure string type
            nunique = self.df[col].nunique(dropna=True)

            try:
                if nunique <= 10:
                    # Safe label encoding for low-cardinality columns
                    le = LabelEncoder()
                    encoded_df[col] = le.fit_transform(self.df[col])
                else:
                    # One-hot encode high-cardinality categoricals
                    one_hot = pd.get_dummies(self.df[col], prefix=col, dtype=float)
                    encoded_df = pd.concat([encoded_df, one_hot], axis=1)
            except Exception as e:
                print(f"Encoding failed for {col}: {e}")
                continue

        # Combine and drop originals
        self.df = pd.concat([self.df, encoded_df], axis=1)
        self.df.drop(columns=self.categorical_features, inplace=True, errors="ignore")

        self.report["categorical_encoded"] = True




    def process_text_columns(self):
        for col in self.text_features:
            vectorizer = TfidfVectorizer(max_features=200)
            vec = vectorizer.fit_transform(self.df[col].astype(str))
            vec_df = pd.DataFrame(
                vec.toarray(),
                columns=[f"{col}_tfidf_{i}" for i in range(vec.shape[1])],
            )
            self.df = pd.concat([self.df, vec_df], axis=1)

        if self.text_features:
            self.df.drop(columns=self.text_features, inplace=True)
        self.report["text_vectorized"] = len(self.text_features) > 0



    def scale_numeric(self):
        scaler = StandardScaler()
        if self.numeric_features:
            numeric_cols = [col for col in self.numeric_features if np.issubdtype(self.df[col].dtype, np.number)]
            self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
        self.report["numeric_scaled"] = len(self.numeric_features) > 0



    def run(self):
        self.inspect_column_types()
        self.detect_column_types()
        self.handle_missing()
        self.encode_categoricals()
        self.process_text_columns()
        self.scale_numeric()
        X = self.df.drop(columns=[self.target], errors="ignore")
        y = self.df[self.target] if self.target in self.df.columns else None
        self.report["final_shape"] = X.shape
        return X, y, self.report

