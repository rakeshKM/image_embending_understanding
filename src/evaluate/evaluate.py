import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def query_image(text_embedding, image_embeddings, top_k=3):
    similarities = cosine_similarity(text_embedding.reshape(1, -1), image_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return top_indices, similarities[top_indices]

def mean_reciprocal_rank(queries, relevant_indices, image_embeddings):
    mrr_sum = 0.0
    
    for text_embedding, relevant in zip(queries, relevant_indices):
        top_indices, _ = query_image(text_embedding, image_embeddings)
        
        rank = next((i + 1 for i, idx in enumerate(top_indices) if idx in relevant), None)
        
        if rank is not None:
            mrr_sum += 1.0 / rank

    mrr = mrr_sum / len(queries) if queries else 0
    return mrr

if __name__ == "__main__":
    # Example queries with corresponding relevant indices
    queries = text_embeddings  # Use the generated text embeddings
    relevant_indices = [
        [0],  # Assume image 1 is relevant for the first query
        [1],  # Assume image 2 is relevant for the second query
        [2]   # Assume image 3 is relevant for the third query
    ]

    # Calculate MRR
    mrr_score = mean_reciprocal_rank(queries, relevant_indices, image_embeddings)

    print("Mean Reciprocal Rank (MRR):", mrr_score)