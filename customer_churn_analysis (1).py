"""
IBM Data Analyst Internship Project
Project: Customer Churn Analysis and Prediction

Author: Pranav Marathe
Domain: Data Analytics / Predictive Analytics
Tools: Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

This script is designed to run even when the original dataset is not available.
If customer_churn.csv is present in the same folder, it will use that file.
Otherwise, it creates a reproducible synthetic telecom-customer dataset for
demonstration and learning purposes.

Outputs are saved inside the "outputs" folder.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# 1. Data loading / fallback dataset generation
# ---------------------------------------------------------
def create_demo_dataset(n=1200):
    """Create a realistic synthetic telecom customer dataset."""
    rng = np.random.default_rng(RANDOM_STATE)

    gender = rng.choice(["Male", "Female"], n)
    senior = rng.choice([0, 1], n, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], n, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], n, p=[0.30, 0.70])
    tenure = rng.integers(1, 73, n)

    internet = rng.choice(
        ["DSL", "Fiber optic", "No internet"],
        n, p=[0.34, 0.46, 0.20]
    )
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        n, p=[0.56, 0.24, 0.20]
    )
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        n, p=[0.34, 0.22, 0.23, 0.21]
    )
    paperless = rng.choice(["Yes", "No"], n, p=[0.62, 0.38])
    tech_support = rng.choice(["Yes", "No"], n, p=[0.28, 0.72])
    online_security = rng.choice(["Yes", "No"], n, p=[0.32, 0.68])

    monthly = (
        20
        + (internet == "Fiber optic") * 42
        + (internet == "DSL") * 24
        + (contract == "Month-to-month") * 7
        + rng.normal(0, 6, n)
    )
    monthly = np.clip(monthly, 18, 120)
    total = monthly * tenure + rng.normal(0, 120, n)
    total = np.clip(total, 20, None)

    # Churn probability intentionally depends on meaningful business factors.
    logit = (
        -1.8
        + 1.25 * (contract == "Month-to-month")
        - 0.028 * tenure
        + 0.70 * (internet == "Fiber optic")
        + 0.45 * (payment == "Electronic check")
        + 0.45 * (tech_support == "No")
        + 0.35 * (online_security == "No")
        + 0.30 * (senior == 1)
        + 0.018 * (monthly - 60)
        + rng.normal(0, 0.45, n)
    )
    probability = 1 / (1 + np.exp(-logit))
    churn = np.where(rng.random(n) < probability, "Yes", "No")

    df = pd.DataFrame({
        "CustomerID": [f"CUST{10000+i}" for i in range(n)],
        "Gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "TenureMonths": tenure,
        "InternetService": internet,
        "Contract": contract,
        "PaymentMethod": payment,
        "PaperlessBilling": paperless,
        "TechSupport": tech_support,
        "OnlineSecurity": online_security,
        "MonthlyCharges": np.round(monthly, 2),
        "TotalCharges": np.round(total, 2),
        "Churn": churn
    })

    # Add a few missing values to demonstrate data cleaning.
    for col in ["TotalCharges", "TechSupport", "OnlineSecurity"]:
        idx = rng.choice(df.index, size=max(1, int(0.02 * n)), replace=False)
        df.loc[idx, col] = np.nan

    return df


def load_data():
    csv_path = BASE_DIR / "customer_churn.csv"
    if csv_path.exists():
        print(f"Using dataset: {csv_path.name}")
        return pd.read_csv(csv_path)

    print("customer_churn.csv not found.")
    print("Creating reproducible demonstration dataset instead.")
    return create_demo_dataset()


# ---------------------------------------------------------
# 2. Data cleaning and preprocessing
# ---------------------------------------------------------
def clean_data(df):
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.replace(" ", "_")
        .str.replace(r"[^\w_]", "", regex=True)
    )

    # Standardize common numeric fields if present.
    for col in ["MonthlyCharges", "TotalCharges", "TenureMonths", "SeniorCitizen"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove duplicate rows.
    df = df.drop_duplicates()

    # Normalize target values.
    if "Churn" in df.columns:
        df["Churn"] = (
            df["Churn"].astype(str).str.strip().str.title()
        )

    return df


# ---------------------------------------------------------
# 3. Exploratory Data Analysis
# ---------------------------------------------------------
def run_eda(df):
    print("\n========== DATA OVERVIEW ==========")
    print("Shape:", df.shape)
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isna().sum())
    print("\nDescriptive statistics:")
    print(df.describe(include="all").T)

    # Save cleaned data snapshot.
    df.to_csv(OUTPUT_DIR / "cleaned_customer_churn.csv", index=False)

    # Churn distribution
    if "Churn" in df.columns:
        churn_rate = (df["Churn"] == "Yes").mean() * 100
        print(f"\nOverall churn rate: {churn_rate:.2f}%")

        plt.figure(figsize=(7, 5))
        sns.countplot(data=df, x="Churn")
        plt.title("Customer Churn Distribution")
        plt.xlabel("Churn")
        plt.ylabel("Number of Customers")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "01_churn_distribution.png", dpi=200)
        plt.close()

    # Contract vs churn
    if {"Contract", "Churn"}.issubset(df.columns):
        contract_churn = pd.crosstab(
            df["Contract"], df["Churn"], normalize="index"
        ) * 100
        print("\nChurn percentage by contract:")
        print(contract_churn.round(2))

        plt.figure(figsize=(9, 5))
        sns.countplot(data=df, x="Contract", hue="Churn")
        plt.title("Churn by Contract Type")
        plt.xticks(rotation=10)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "02_contract_vs_churn.png", dpi=200)
        plt.close()

    # Tenure distribution
    if {"TenureMonths", "Churn"}.issubset(df.columns):
        plt.figure(figsize=(9, 5))
        sns.histplot(
            data=df, x="TenureMonths", hue="Churn",
            bins=24, kde=True, element="step"
        )
        plt.title("Tenure Distribution by Churn")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "03_tenure_vs_churn.png", dpi=200)
        plt.close()

    # Monthly charges
    if {"MonthlyCharges", "Churn"}.issubset(df.columns):
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
        plt.title("Monthly Charges vs Churn")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "04_monthly_charges_vs_churn.png", dpi=200)
        plt.close()

    # Correlation heatmap for numeric columns
    numeric_df = df.select_dtypes(include=np.number)
    if not numeric_df.empty:
        plt.figure(figsize=(9, 6))
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "05_correlation_heatmap.png", dpi=200)
        plt.close()


# ---------------------------------------------------------
# 4. Predictive analytics
# ---------------------------------------------------------
def prepare_ml_data(df):
    data = df.copy()

    target = "Churn"
    if target not in data.columns:
        raise ValueError("Target column 'Churn' is required.")

    data[target] = data[target].map({"Yes": 1, "No": 0})

    # CustomerID is an identifier, not a useful predictor.
    X = data.drop(columns=[target, "CustomerID"], errors="ignore")
    y = data[target]

    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = X.select_dtypes(include=np.number).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical)
    ])

    return X, y, preprocessor


def evaluate_model(model_name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probabilities)
    else:
        auc = np.nan

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1_Score": f1_score(y_test, predictions, zero_division=0),
        "ROC_AUC": auc
    }

    print(f"\n========== {model_name} ==========")
    print(pd.Series(metrics))
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"]
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    plt.savefig(OUTPUT_DIR / f"06_confusion_{safe_name}.png", dpi=200)
    plt.close()

    return model, metrics


def run_predictive_analytics(df):
    X, y, preprocessor = prepare_ml_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    logistic = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
    ])

    tree = Pipeline([
        ("preprocessor", preprocessor),
        ("model", DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=10, random_state=RANDOM_STATE
        ))
    ])

    models = []
    for name, model in [
        ("Logistic Regression", logistic),
        ("Decision Tree", tree)
    ]:
        fitted, metrics = evaluate_model(
            name, model, X_train, X_test, y_train, y_test
        )
        models.append((fitted, metrics))

    results = pd.DataFrame([m[1] for m in models])
    results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    print("\nModel comparison:")
    print(results.round(4).to_string(index=False))

    return models, X_test, y_test


# ---------------------------------------------------------
# 5. Customer segmentation using K-Means
# ---------------------------------------------------------
def run_customer_segmentation(df):
    required = ["TenureMonths", "MonthlyCharges", "TotalCharges"]
    if not all(col in df.columns for col in required):
        print("\nSegmentation skipped: required numeric columns missing.")
        return

    segment_df = df[required].copy()
    segment_df = segment_df.fillna(segment_df.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(segment_df)

    scores = {}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        scores[k] = silhouette_score(X_scaled, labels)

    best_k = max(scores, key=scores.get)
    final_model = KMeans(
        n_clusters=best_k, random_state=RANDOM_STATE, n_init=10
    )
    labels = final_model.fit_predict(X_scaled)

    segment_df["Cluster"] = labels
    segment_df.to_csv(OUTPUT_DIR / "customer_segments.csv", index=False)

    print("\n========== CUSTOMER SEGMENTATION ==========")
    print("Silhouette scores:", {k: round(v, 3) for k, v in scores.items()})
    print("Selected clusters:", best_k)
    print("\nCluster summary:")
    print(segment_df.groupby("Cluster").mean(numeric_only=True).round(2))

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=segment_df,
        x="TenureMonths",
        y="MonthlyCharges",
        hue="Cluster",
        palette="viridis",
        s=70
    )
    plt.title("Customer Segmentation")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "07_customer_segmentation.png", dpi=200)
    plt.close()


# ---------------------------------------------------------
# 6. Business insights
# ---------------------------------------------------------
def generate_business_insights(df):
    insights = []

    if "Churn" in df.columns:
        churn_rate = (df["Churn"] == "Yes").mean() * 100
        insights.append(f"Overall customer churn in the analyzed data is approximately {churn_rate:.1f}%.")

    if {"Contract", "Churn"}.issubset(df.columns):
        rates = (
            df.assign(ChurnFlag=(df["Churn"] == "Yes").astype(int))
            .groupby("Contract")["ChurnFlag"]
            .mean()
            .sort_values(ascending=False)
        )
        if not rates.empty:
            top_contract = rates.index[0]
            insights.append(
                f"The {top_contract} contract group has the highest observed churn rate "
                f"among contract categories."
            )

    if {"PaymentMethod", "Churn"}.issubset(df.columns):
        rates = (
            df.assign(ChurnFlag=(df["Churn"] == "Yes").astype(int))
            .groupby("PaymentMethod")["ChurnFlag"]
            .mean()
            .sort_values(ascending=False)
        )
        if not rates.empty:
            insights.append(
                f"Churn varies across payment methods; {rates.index[0]} shows the highest "
                f"observed churn rate in this analysis."
            )

    if {"TechSupport", "Churn"}.issubset(df.columns):
        rates = (
            df.assign(ChurnFlag=(df["Churn"] == "Yes").astype(int))
            .groupby("TechSupport")["ChurnFlag"]
            .mean()
        )
        if "No" in rates and "Yes" in rates:
            if rates["No"] > rates["Yes"]:
                insights.append(
                    "Customers without technical support show a higher observed churn rate "
                    "than customers with technical support."
                )

    if {"TenureMonths", "Churn"}.issubset(df.columns):
        avg_tenure = df.groupby("Churn")["TenureMonths"].mean()
        if "Yes" in avg_tenure and "No" in avg_tenure:
            insights.append(
                f"Average tenure differs between churned and retained customers "
                f"({avg_tenure['Yes']:.1f} vs {avg_tenure['No']:.1f} months)."
            )

    output = "\n".join(f"{i+1}. {x}" for i, x in enumerate(insights))
    (OUTPUT_DIR / "business_insights.txt").write_text(output, encoding="utf-8")

    print("\n========== BUSINESS INSIGHTS ==========")
    print(output)


# ---------------------------------------------------------
# 7. Main execution
# ---------------------------------------------------------
def main():
    print("=" * 70)
    print("IBM DATA ANALYST INTERNSHIP PROJECT")
    print("Customer Churn Analysis and Prediction")
    print("=" * 70)

    df = load_data()
    df = clean_data(df)

    run_eda(df)
    run_predictive_analytics(df)
    run_customer_segmentation(df)
    generate_business_insights(df)

    print("\nProject completed successfully.")
    print(f"All output files are available in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
