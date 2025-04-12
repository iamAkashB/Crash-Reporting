
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

### Read the dataset Crash-Driving_Dataset
car_dt = pd.read_csv("CrashRepotingDataset.csv")
car_dt.head(10)

### Handling the Missing Values in the dataset
car_dt = car_dt.dropna(thresh = len(car_dt) * 0.2, axis = 1)
car_dt.head(10)

### Fill missing values with mode (for categorical) and median (for numeric)
# Separate numerical and categorical columns
numerical_cols = car_dt.select_dtypes(include=["number"]).columns
categorical_cols = car_dt.select_dtypes(include=["object"]).columns
# Fill missing values
car_dt[numerical_cols] = car_dt[numerical_cols].apply(lambda x: x.fillna(x.median()))  # Median for numerical
car_dt[categorical_cols] = car_dt[categorical_cols].apply(lambda x: x.fillna(x.mode()[0]))  # Mode for categorical
car_dt

# Remove unrealistic vehicle years (valid range: 1900-2025)
car_dt["Vehicle Year"] = car_dt["Vehicle Year"].apply(lambda x: x if 1900 <= x <= 2025 else None)

# Ensure speed limits are within a valid range (0-75 mph)
car_dt = car_dt[(car_dt["Speed Limit"] > 0) & (car_dt["Speed Limit"] <= 75)]
car_dt

# Convert date column to datetime format
# Convert "Crash Date/Time" to datetime with a specific format
car_dt["Crash Date/Time"] = pd.to_datetime(car_dt["Crash Date/Time"], format="%m/%d/%Y %H:%M", errors="coerce")
car_dt


car_dt.drop_duplicates(inplace=True)
car_dt

car_dt.to_csv("Cleaned_Crash_Data.csv", index=False)
print("Data cleaning complete. File saved as 'Cleaned_Crash_Data.csv'.")
car_dt

### A heatmap of correaltion is typically used to identify patterns and relationship and potensial pattern redundancy among numeric feature..
plt.figure(figsize = (12,8))
sns.heatmap(car_dt.corr(numeric_only = True), annot = True, cmap = 'coolwarm', linewidths = 0.5)
plt.title("Correlation Heatmap")
plt.show()


sns.countplot(x='Injury Severity', data=car_dt)
plt.title('Distribution of Injury Severity (Crash Severity)', pad = 20)
plt.xticks(rotation=45)
plt.show()
plt.tight_layout()


car_dt.hist(figsize=(15, 10), bins=20)
plt.tight_layout()
plt.show()

# Clean column names
car_dt.columns = car_dt.columns.str.strip()

# Create the boxplot
sns.boxplot(x='Injury Severity', y='Speed Limit', data=car_dt)
plt.title('Speed Limit Distribution Across Injury Severity')
plt.xticks(rotation=45)
plt.show()

sns.countplot(x='Surface Condition', data=car_dt)
plt.title('Crashes by Road Surface Condition')
plt.xticks(rotation=45)
plt.show()


sample_df = car_dt[['Speed Limit', 'Vehicle Year', 'Injury Severity']].dropna().sample(n=500)  # sample to avoid slow plots
sns.pairplot(sample_df, hue='Injury Severity')
plt.show()


car_dt['Crash Date/Time'] = pd.to_datetime(car_dt['Crash Date/Time'], errors='coerce')

# Daily crash count
car_dt['Crash Date'] = car_dt['Crash Date/Time'].dt.date
daily_crashes = car_dt.groupby('Crash Date').size()

plt.figure(figsize=(12, 6))
daily_crashes.plot()
plt.title('Daily Crash Counts Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Crashes')
plt.grid(True)
plt.show()

car_dt['Injury Severity'].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette('Set2'), startangle=90)
plt.title('Injury Severity Distribution')
plt.ylabel('')
plt.show()



























