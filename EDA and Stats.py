import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import warnings

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, mean_squared_error
from pandas_profiling import ProfileReport

warnings.filterwarnings('ignore')

# Setup output directory
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load file
def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found: {file_path}")
    df = pd.read_csv(file_path)
    print(f"✅ Loaded: {file_path}")
    return df

# EDA and HTML profiling
def perform_eda(df):
    print("\n📊 Running EDA report...")
    profile = ProfileReport(df, title="EDA Report", explorative=True)
    profile.to_file(os.path.join(OUTPUT_DIR, "eda_report.html"))
    print("✅ EDA HTML report saved as 'eda_report.html'")

# Correlation heatmap
def show_correlations(df):
    numeric_df = df.select_dtypes(include=np.number)
    corr = numeric_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.tight_layout()
    corr_path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    plt.savefig(corr_path)
    plt.close()
    print(f"✅ Correlation heatmap saved as '{corr_path}'")
    return corr

# KMeans
def apply_kmeans(df, n_clusters=3):
    numeric_df = df.select_dtypes(include=np.number).dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric_df)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(scaled)
    
    df['Cluster'] = np.nan
    df.loc[numeric_df.index, 'Cluster'] = labels
    print("✅ KMeans Clustering applied.")
    return df

# Encode buyer column
def encode_target(df):
    if 'buyer' not in df.columns:
        raise ValueError("❌ Target column 'buyer' not found.")
    df['buyer'] = df['buyer'].map({'yes': 1, 'no': 0})
    print("✅ Encoded 'buyer' column.")
    return df

# Train and evaluate models
def train_models(df):
    df = encode_target(df)
    df = df.dropna(subset=['buyer'])

    X = df.drop(columns=['buyer'])
    X = X.select_dtypes(include=np.number)
    y = df['buyer']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Results output
    with open(os.path.join(OUTPUT_DIR, "model_summary.txt"), "w") as f:

        # Logistic Regression
        if y.nunique() == 2:
            print("\n🤖 Training Logistic Regression...")
            logreg = LogisticRegression(max_iter=1000)
            logreg.fit(X_train, y_train)
            y_pred_log = logreg.predict(X_test)
            log_report = classification_report(y_test, y_pred_log)
            print(log_report)
            f.write("=== Logistic Regression Report ===\n")
            f.write(log_report + "\n")

            # SHAP for Logistic
            explainer = shap.Explainer(logreg, X_train)
            shap_values = explainer(X_test)
            shap_html_path = os.path.join(OUTPUT_DIR, "shap_logistic.html")
            shap.save_html(shap_html_path, shap.plots.beeswarm(shap_values, show=False))
            print(f"✅ SHAP values (Logistic) saved as '{shap_html_path}'")
        
        # Linear Regression
        print("\n🤖 Training Linear Regression...")
        linreg = LinearRegression()
        linreg.fit(X_train, y_train)
        y_pred_lin = linreg.predict(X_test)
        mse = mean_squared_error(y_test, y_pred_lin)
        f.write("\n=== Linear Regression MSE ===\n")
        f.write(f"MSE: {mse:.4f}\n")
        print(f"📉 Linear Regression MSE: {mse:.4f}")

    print("✅ Model performance written to 'model_summary.txt'")

# Main
def main():
    file_path = r"C:\Users\SEBBIE\Documents\AustraliaGWI\buyer_data.csv"
    df = load_data(file_path)
    perform_eda(df)
    show_correlations(df)
    df = apply_kmeans(df)
    train_models(df)

if __name__ == "__main__":
    main()
