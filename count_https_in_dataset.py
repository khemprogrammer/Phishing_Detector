import pandas as pd
import os

file_path_phiusiil = r"c:\Users\kheml\OneDrive\Desktop\Django Project\Phishing_Website_Detection\ml\data\PhiUSIIL_Phishing_URL_Dataset.csv"
file_path_phishing = r"c:\Users\kheml\OneDrive\Desktop\Django Project\Phishing_Website_Detection\ml\data\dataset_phishing.csv"

df_combined = pd.DataFrame()

try:
    if os.path.exists(file_path_phiusiil):
        df_phiusiil = pd.read_csv(file_path_phiusiil)
        df_phiusiil = df_phiusiil.rename(columns={"URL": "url", "label": "original_label"})
        df_phiusiil["label"] = df_phiusiil["original_label"].map({1: 0, 0: 1}) # Invert labels: 1 (legitimate) -\u003e 0, 0 (phishing) -\u003e 1
        df_phiusiil = df_phiusiil.dropna(subset=["url", "label"]).loc[:, ["url", "label"]]
        df_combined = pd.concat([df_combined, df_phiusiil], ignore_index=True)
        print(f"Loaded PhiUSIIL_Phishing_URL_Dataset.csv with {len(df_phiusiil)} entries.")

    if os.path.exists(file_path_phishing):
        df_phishing = pd.read_csv(file_path_phishing)
        if 'status' in df_phishing.columns:
            df_phishing['label'] = df_phishing['status'].map({'phishing': 1, 'legitimate': 0})
        elif 'label' not in df_phishing.columns:
            raise KeyError("Neither 'status' nor 'label' column found in dataset_phishing.csv.")
        df_phishing = df_phishing.dropna(subset=["url","label"]).loc[:, ["url","label"]]
        df_combined = pd.concat([df_combined, df_phishing], ignore_index=True)
        print(f"Loaded dataset_phishing.csv with {len(df_phishing)} entries.")

    if df_combined.empty:
        raise FileNotFoundError("No dataset found at specified paths.")

    df_combined = df_combined.drop_duplicates(subset=["url"]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    df_combined['has_https'] = df_combined["url"].apply(lambda x: 1 if isinstance(x, str) and x.lower().startswith("https://") else 0)

    print(f"\nAnalysis of combined dataset (Total entries: {len(df_combined)}):")
    print("\nDistribution of 'has_https' for each label:")
    print(df_combined.groupby('label')['has_https'].value_counts(normalize=True) * 100)

    print("\nTotal HTTPS URLs:")
    print(df_combined['has_https'].sum())

except FileNotFoundError as e:
    print(f"Error: {e}")
except KeyError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")