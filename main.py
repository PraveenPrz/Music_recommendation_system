# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load the dataset
df = pd.read_csv('C:\Dataset2\data.csv')  # Replace 'your_dataset.csv' with the actual file path

# Display the first few rows of the dataset
print(df.head())

# Summary statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Define numerical features
numerical_features = ['popularity', 'duration_ms', 'explicit', 'danceability', 'energy', 'key',
                      'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
                      'liveness', 'valence', 'tempo', 'time_signature']

# Sample a subset of data
df_sample = df.sample(frac=0.2)  # Adjust the fraction as needed

# Pairplot for numerical features using sampled data
sns.pairplot(df_sample[numerical_features])
plt.suptitle('Pairplot of Numerical Features (Sampled Data)')
plt.show()

# Boxplot for popularity vs. genre using sampled data
plt.figure(figsize=(15, 8))
sns.boxplot(x='track_genre', y='popularity', data=df_sample)
plt.title('Boxplot of Popularity by Genre (Sampled Data)')
plt.xticks(rotation=45)
plt.show()

# Scatter plot of energy vs. danceability for a specific genre using sampled data
genre_to_plot = 'acoustic'  # Replace with the desired genre
df_genre = df_sample[df_sample['track_genre'] == genre_to_plot]

fig = px.scatter(df_genre, x='energy', y='danceability', color='track_genre',
                 size='popularity', hover_name='track_name',
                 title=f'Scatter Plot of Energy vs. Danceability for {genre_to_plot} Genre (Sampled Data)')
fig.show()

# Violin plot for explicitness and loudness using sampled data
plt.figure(figsize=(15, 8))
sns.violinplot(x='explicit', y='loudness', data=df_sample)
plt.title('Violin Plot of Explicitness and Loudness (Sampled Data)')
plt.show()

# Bar plot for average popularity by key using sampled data
average_popularity_by_key = df_sample.groupby('key')['popularity'].mean().reset_index()
plt.figure(figsize=(12, 6))
sns.barplot(x='key', y='popularity', data=average_popularity_by_key, palette='viridis')
plt.title('Average Popularity by Key (Sampled Data)')
plt.show()

# Pie chart for the distribution of time signatures using sampled data
time_signature_counts = df_sample['time_signature'].value_counts()
fig, ax = plt.subplots()
ax.pie(time_signature_counts, labels=time_signature_counts.index, autopct='%1.1f%%', startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that the pie is drawn as a circle.
plt.title('Distribution of Time Signatures (Sampled Data)')
plt.show()
