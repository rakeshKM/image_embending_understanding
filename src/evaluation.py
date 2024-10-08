import torch

def mean_reciprocal_rank(y_true, y_pred):
    """
    y_true: list of ground truth labels
    y_pred: list of predictions (ranked lists of items)
    """
    rr_sum = 0
    for true, pred in zip(y_true, y_pred):
        try:
            rank = pred.index(true) + 1
            rr_sum += 1.0 / rank
        except ValueError:
            pass  # If the true label is not found in the prediction list
    return rr_sum / len(y_true)

def compute_mrr(image_embeddings, text_embeddings, top_k=10):
    """
    Compare image embeddings to text embeddings using cosine similarity
    and compute the MRR.
    """
    similarities = torch.matmul(image_embeddings, text_embeddings.T)
    rankings = torch.argsort(similarities, dim=1, descending=True)
    
    ground_truth = torch.arange(image_embeddings.shape[0])
    pred_ranks = []

    for i in range(similarities.shape[0]):
        ranked_indices = rankings[i][:top_k].tolist()
        pred_ranks.append(ranked_indices)

    mrr = mean_reciprocal_rank(ground_truth.tolist(), pred_ranks)
    return mrr
