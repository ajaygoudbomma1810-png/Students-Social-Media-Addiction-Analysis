from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import sys

# ------------------------------------------
# 1. Set up paths
# ------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW = DATA_DIR / "Students Social Media Addiction - Copy.csv"   # <-- YOUR FILE NAME

print("Looking for CSV at:", RAW)

if not RAW.exists():
    print("\n❌ ERROR: CSV file not found!")
    print("Make sure the file is inside:", DATA_DIR)
    print("Files inside data/:", list(DATA_DIR.glob('*')))
    sys.exit(1)

# ------------------------------------------
# 2. Load Data
# ------------------------------------------
df = pd.read_csv(RAW)
print("\n✅ File loaded successfully!")
print("Rows:", df.shape[0], "Columns:", df.shape[1])

# ------------------------------------------
# 3. Basic Cleaning
# ------------------------------------------

# Remove exact duplicates
df.drop_duplicates(inplace=True)

# Fix column name spacing issues
df.columns = df.columns.str.strip()

# Remove negative values (not logical)
num_cols = ['Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Mental_Health_Score', 'Addicted_Score']
for col in num_cols:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: np.nan if x < 0 else x)

# Fill missing values (median for numeric)
for col in num_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# Clean text columns
text_cols = ['Gender', 'Most_Used_Platform', 'Academic_Level', 'Country']
for col in text_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.title()

# ------------------------------------------
# 4. Feature Engineering
# ------------------------------------------

# Sleep adjustment factor
df['Sleep_Adjustment'] = df['Sleep_Hours_Per_Night'].apply(
    lambda x: -5 if x < 6 else (5 if x > 8 else 0)
)

# Compute addiction score
df['computed_addiction_score'] = (
    df['Avg_Daily_Usage_Hours'] * 2 +
    df['Addicted_Score'] * 1.5 +
    df['Sleep_Adjustment']
)

# Categorize addiction level
df['addiction_level'] = df['computed_addiction_score'].apply(
    lambda x: 'High' if x > 25 else ('Medium' if x > 15 else 'Low')
)

# ------------------------------------------
# 5. Aggregations for Power BI
# ------------------------------------------

# Academic level insights
agg_academic = df.groupby('Academic_Level').agg(
    avg_addiction=('computed_addiction_score', 'mean'),
    pct_high_addiction=('addiction_level', lambda x: (x == 'High').mean() * 100),
    avg_sleep=('Sleep_Hours_Per_Night', 'mean'),
    avg_mental_health=('Mental_Health_Score', 'mean'),
    count=('Student_ID', 'count')
).reset_index()

# Platform-based insights
agg_platform = df.groupby('Most_Used_Platform').agg(
    avg_addiction=('computed_addiction_score', 'mean'),
    pct_high_addiction=('addiction_level', lambda x: (x == 'High').mean() * 100),
    count=('Student_ID', 'count')
).reset_index().sort_values(by='count', ascending=False)

# ------------------------------------------
# 6. Clustering for segmentation
# ------------------------------------------
features = df[['computed_addiction_score', 'Sleep_Hours_Per_Night', 'Mental_Health_Score']].fillna(0)

kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(features).astype(str)

# ------------------------------------------
# 7. Save outputs
# ------------------------------------------
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

clean_path = OUTPUT_DIR / "students_cleaned.csv"
agg_academic_path = OUTPUT_DIR / "agg_by_academic_level.csv"
agg_platform_path = OUTPUT_DIR / "agg_by_platform.csv"
cluster_path = OUTPUT_DIR / "student_clusters.csv"

df.to_csv(clean_path, index=False)
agg_academic.to_csv(agg_academic_path, index=False)
agg_platform.to_csv(agg_platform_path, index=False)
df[['Student_ID', 'cluster', 'computed_addiction_score']].to_csv(cluster_path, index=False)

# ------------------------------------------
# 8. Done
# ------------------------------------------
print("\n🎉 Pipeline completed successfully!")
print("Saved outputs in:", OUTPUT_DIR)
print(" - students_cleaned.csv")
print(" - agg_by_academic_level.csv")
print(" - agg_by_platform.csv")
print(" - student_clusters.csv")
