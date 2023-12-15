import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the dataset
# Assuming 'data.csv' is your dataset file
data = pd.read_csv('C:\\Dataset2\\data.csv')

# Extract features for clustering
features = data[['duration_ms', 'explicit', 'danceability', 'energy', 'key', 'loudness', 'mode',
                 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence',
                 'time_signature']]

# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Determine optimal number of clusters using silhouette score
max_clusters = 10  # You can adjust this based on your preference
best_score = -1
best_clusters = 2

for clusters in range(2, max_clusters + 1):
    kmeans = KMeans(n_clusters=clusters, random_state=42, n_init=10)  # Explicitly set n_init
    cluster_labels = kmeans.fit_predict(scaled_features)
    score = silhouette_score(scaled_features, cluster_labels)

    if score > best_score:
        best_score = score
        best_clusters = clusters

# Fit KMeans with optimal number of clusters
kmeans = KMeans(n_clusters=best_clusters, random_state=42, n_init=10)  # Explicitly set n_init
data['cluster'] = kmeans.fit_predict(scaled_features)


# Function to recommend top songs in the cluster of the user-inputted song
def recommend_songs_for_user_input(song_name):
    # Find the cluster of the user-inputted song
    user_song = data[data['track_name'].str.lower() == song_name.lower()].iloc[0]
    user_cluster = user_song['cluster']

    # Extract top songs from the user's cluster
    cluster_data = data[data['cluster'] == user_cluster]
    top_songs = cluster_data.sort_values(by='valence', ascending=False).head(10)

    print(f"Top 10 songs in the cluster of '{song_name}':\n")
    for i, row in top_songs.iterrows():
        print(f"{i + 1}. {row['track_name']} - {row['artists']}")

# Get user input for the song name
user_input_song = input("Enter a song name: ")
recommend_songs_for_user_input(user_input_song)
