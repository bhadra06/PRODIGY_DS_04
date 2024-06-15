import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
file_path = 'twitter_entity_sentiment_analysis.csv'

# Load data
df = pd.read_csv(file_path)

# Step 2: Print the first few rows and column names for verification
print("Column names in the dataset:")
print(df.columns)
print("\nFirst few rows of the dataset:")
print(df.head())

# Step 3: Clean column names (if necessary)
df.columns = df.columns.str.strip()  # Remove any leading or trailing whitespace

# Step 4: Check if the required columns exist
required_columns = ['tweet_id', 'entity', 'sentiment', 'text']
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    print(f"Missing columns: {missing_columns}")
else:
    print("All required columns are present.")

# Step 5: Plot sentiment distribution if 'sentiment' column exists
if 'Positive' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Positive', data=df, palette='viridis')
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("Column 'Positive' not found in the dataset.")

# Step 6: Plot sentiment trends over time (if 'timestamp' and 'sentiment_score' columns exist)
if 'timestamp' in df.columns and 'sentiment_score' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])  # Convert timestamp to datetime
    plt.figure(figsize=(14, 7))
    sns.lineplot(x='timestamp', y='sentiment_score', data=df, hue='sentiment')
    plt.title('Sentiment Trends Over Time')
    plt.xlabel('Time')
    plt.ylabel('Sentiment Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("Timestamp or sentiment_score columns not found. Skipping sentiment trends visualization.")
