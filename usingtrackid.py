import pyarrow.parquet as pq
import duckdb
import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import KNNBasic
from surprise import accuracy

# Load data from Parquet file
parquet_file_path = "C:\\Dataset1\\"  # Adjust the filename and use double backslashes
table = pq.read_table(parquet_file_path)
df_from_parquet = table.to_pandas()

# Display the DataFrame from Parquet
print("DataFrame from Parquet:")
print(df_from_parquet.head())

# Load data from DuckDB
duckdb_file_path = "C:\\Dataset2\\Index1.duckdb"  # Adjust the filename and use double backslashes
connection = duckdb.connect(database=duckdb_file_path, read_only=True)

# Get a list of tables in the database
tables = connection.execute("SELECT table_name FROM information_schema.tables").fetchdf()
print("\nTables in the DuckDB database:")
print(tables)

# Use the first table in the query
if not tables.empty:
    actual_table_name = tables['table_name'][0]
    # Modify the query to include the schema name
    df_from_duckdb = connection.execute(f"SELECT * FROM fts_main_data.{actual_table_name}").fetchdf()

    # Display the DataFrame from DuckDB
    print(f"\nDataFrame from DuckDB (Table: {actual_table_name}):")
    print(df_from_duckdb.head())
else:
    print("No tables found in the DuckDB database.")

# Recommendation System
# Load the relevant columns from the Parquet DataFrame
df_for_recommendation = df_from_parquet[['track_id', 'popularity', 'danceability', 'energy', 'valence', 'tempo', 'track_genre']]

# Create a synthetic 'rating' column (adjust the formula based on your preference)
df_for_recommendation['rating'] = df_for_recommendation['popularity'] / 10.0

# Create a synthetic 'user_id' column for training
df_for_recommendation['user_id'] = 0

# Load data into the Surprise library format
reader = Reader(rating_scale=(0, 100))
data = Dataset.load_from_df(df_for_recommendation[['user_id', 'track_id', 'rating']], reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Use item-based collaborative filtering
algo = KNNBasic(sim_options={'user_based': False})

# Train the algorithm on the training set
algo.fit(trainset)

# Function to get recommendations for the next song based on the previous song
def get_next_song_recommendation(previous_track_id, num_recommendations=5):
    dummy_user_id = 0
    anti_testset = [(dummy_user_id, trainset.to_inner_iid(previous_track_id), 0) for _ in range(num_recommendations)]

    predictions = algo.test(anti_testset)
    recommendations = sorted(predictions, key=lambda x: x.est, reverse=True)[:num_recommendations]

    recommended_tracks = [(trainset.to_raw_iid(pred.iid), pred.est) for pred in recommendations]

    return recommended_tracks

# Example: Get recommendations for the next song based on a previous song
previous_track_id = 'previous_track_id'  # Replace with the actual previous track ID
next_song_recommendations = get_next_song_recommendation(previous_track_id)
print(f"\nTop {len(next_song_recommendations)} recommendations for the next song after {previous_track_id}:")
for track_id, estimated_rating in next_song_recommendations:
    print(f"Track ID: {track_id}, Estimated Rating: {estimated_rating}")
