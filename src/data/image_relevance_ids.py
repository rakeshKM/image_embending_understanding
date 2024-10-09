import pandas as pd

# Load your data (assuming it's in a CSV file)
CSV_FILE = "/data/rakesh/vision_pod/public_datasets/Car_parts/image_labels.csv"

# Read the CSV file
data = pd.read_csv(CSV_FILE)

# Assuming the CSV has columns 'image_path' and 'group_label'
# Create a dictionary to group indices by their labels
grouped_indices = {}

# Populate the dictionary with indices
for index, row in data.iterrows():
    group_label = row['label_group']
    
    if group_label not in grouped_indices:
        grouped_indices[group_label] = []
    grouped_indices[group_label].append(index)

# Now create a relevance index for each image
relevance_indices = {}

for index, row in data.iterrows():
    group_label = row['label_group']
    
    # Get the relevant indices for this image based on the group label
    relevance_indices[index] = grouped_indices[group_label]

# Print the relevance indices
for image_index, relevant_indices in relevance_indices.items():
    print(f"Image Index: {image_index} | Relevant Indices: {relevant_indices}")