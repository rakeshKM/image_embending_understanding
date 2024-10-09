import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def query_image(text_embedding, image_embeddings, top_k=5):
    # Reshape text_embedding to 2D
    text_embedding_reshaped = text_embedding.reshape(1, -1)
    
    # print("Text embedding reshaped:", text_embedding_reshaped.shape)
    # print("Image embeddings shape:", image_embeddings.shape)

    # Calculate cosine similarity
    similarities = cosine_similarity(text_embedding_reshaped, image_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return top_indices, similarities[top_indices]

def mean_reciprocal_rank(queries, image_embeddings, relevant_indices):
    mrr_sum = 0.0

    for query_idx, text_embedding in enumerate(queries):
        # Get the relevant index for the current query
        relevant_index = relevant_indices[query_idx]
        
        # Query the image embeddings using the provided text embedding
        top_indices,_ = query_image(text_embedding, image_embeddings)
        #print("top_indices",top_indices)
        
        # Check if any of the top retrieved indices match the relevant index
        rank = next((i + 1 for i, idx in enumerate(top_indices) if relevant_index in relevant_indices[idx] ), None)

        if rank is not None:
            mrr_sum += 1.0 / rank

    mrr = mrr_sum / len(queries) if len(queries) else 0
    return mrr

if __name__ == "__main__":
    # Example Usage:
    # Assuming you have the following variables defined:
    # - image_embeddings: numpy array of image embeddings
    # - queries: numpy array of text embeddings for the queries
    # - relevant_indices: list of unique IDs corresponding to each image and corresponding text
    
    # Calculate MRR
    mrr_score = mean_reciprocal_rank(queries, relevant_indices, image_embeddings)

    print("Mean Reciprocal Rank (MRR):", mrr_score)