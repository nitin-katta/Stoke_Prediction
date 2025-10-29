import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from imblearn.over_sampling import RandomOverSampler


# =====================================================
# 1️⃣ Custom Transformers
# =====================================================
class fillna(BaseEstimator, TransformerMixin):
    def __init__(self, column_name='smoking_status', fill_value="Unknown"):
        self.column_name = column_name
        self.fill_value = fill_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        if self.column_name in X_.columns:
            X_[self.column_name] = X_[self.column_name].fillna(self.fill_value)
        return X_


class drop_null_values(BaseEstimator, TransformerMixin):
    def __init__(self, column_name=None):
        self.column_name = column_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        if self.column_name:
            X_ = X_[X_[self.column_name].notnull()]
        else:
            X_ = X_.dropna()
        return X_


class drop_gender_other(BaseEstimator, TransformerMixin):
    def __init__(self, column_name='gender', drop_value="Other"):
        self.column_name = column_name
        self.drop_value = drop_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        if self.column_name in X_.columns:
            X_ = X_[X_[self.column_name] != self.drop_value]
        return X_


# =====================================================
# 2️⃣ Data Cleaning Pipeline
# =====================================================
handling_data = Pipeline([
    ("fill_missing_smoking", fillna()),
    ("drop_nulls", drop_null_values()),
    ("drop_gender_other", drop_gender_other())
])


# =====================================================
# 3️⃣ Preprocessing (Scaling + Encoding)
# =====================================================
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
numerical_columns = ['bmi', 'avg_glucose_level']

pre_processor = ColumnTransformer([
    ("num", StandardScaler(), numerical_columns),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
])


# =====================================================
# 4️⃣ Model Definition
# =====================================================
model = RandomForestClassifier(random_state=101)


# =====================================================
# 5️⃣ Full Pipeline Assembly
# =====================================================
def build_pipeline():
    """Builds the complete model training pipeline."""
    final_pipeline = Pipeline([
        ("preprocessor", pre_processor),
        ("model", model)
    ])
    return final_pipeline


# =====================================================
# 6️⃣ Main Execution (Load Data Last)
# =====================================================
if __name__ == "__main__":
    print("📂 Loading dataset...")
    data = pd.read_csv("dataset.csv")

    # Data cleaning
    print("🧹 Cleaning data...")
    data_cleaned = handling_data.fit_transform(data)

    # Split target
    X = data_cleaned.drop("stroke", axis=1)
    y = data_cleaned["stroke"]

    # Oversampling minority class
    print("🔄 Applying oversampling...")
    ros = RandomOverSampler(random_state=42, sampling_strategy=0.5)
    X_res, y_res = ros.fit_resample(X, y)

    # Build pipeline
    final_pipeline = build_pipeline()

    # Train model
    print("🚀 Training model...")
    final_pipeline.fit(X_res, y_res)
    print("✅ Model training complete!")

    # Save model
    joblib.dump(final_pipeline, "stroke_model.pkl")
    print("💾 Model saved as stroke_model.pkl")

    # Sample predictions
    sample_preds = final_pipeline.predict(X.head(5))
    print("🧠 Sample Predictions:", sample_preds)

    # Cross-validation (on resampled data)
    print("📊 Running cross-validation...")
    cv_results = cross_validate(final_pipeline, X_res, y_res, cv=5, scoring=['accuracy', 'precision', 'recall'])

    print("\n📈 Model Performance (5-Fold CV):")
    print(f"Mean Accuracy : {cv_results['test_accuracy'].mean():.4f}")
    print(f"Mean Precision: {cv_results['test_precision'].mean():.4f}")
    print(f"Mean Recall   : {cv_results['test_recall'].mean():.4f}")
