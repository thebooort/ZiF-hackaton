
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')  


dict1 = {'country': 'Kenya', 'crop': 'Coffee', 'ecosystem': 'Desert', 'agriculture': 'Organic'}
dict2 = {'country': 'Colombia', 'crop': 'Coffee', 'ecosystem': 'Desert', 'agriculture': 'Organic'}

weights = {
    'country': 2,        # Lower weight
    'crop': 3,           # Higher weight
    'ecosystem': 3,      # Higher weight
    'agriculture': 1     
}

similarity_scores = []
total_weight = sum(weights.values())

for key in dict1:
    if key in dict2:
        # Get embeddings for the values
        embedding1 = model.encode(dict1[key], convert_to_tensor=True)
        embedding2 = model.encode(dict2[key], convert_to_tensor=True)

        # Compute cosine similarity
        cosine_sim = cosine_similarity(
            embedding1.detach().numpy().reshape(1, -1),
            embedding2.detach().numpy().reshape(1, -1)
        )[0][0]


        weighted_similarity = weights[key] * cosine_sim
        similarity_scores.append(weighted_similarity)

final_similarity = sum(similarity_scores) / total_weight

print(f"Similarity Score: {final_similarity:.2f}")




# Function to calculate similarity for a single row
def compute_similarity(row, reference_dict, weights, ollama):
    similarity_scores = []
    total_weight = sum(weights.values())
    
    for key in reference_dict.keys():
        # Embed the row and reference values using Ollama
        embedding1 = ollama.embed_query(str(row[key]))
        embedding2 = ollama.embed_query(str(reference_dict[key]))
        
        # Compute cosine similarity
        cosine_sim = cosine_similarity(
            np.array(embedding1).reshape(1, -1),
            np.array(embedding2).reshape(1, -1)
        )[0][0]
        
        # Weight the similarity
        weighted_similarity = weights[key] * cosine_sim
        similarity_scores.append(weighted_similarity)
    
    # Normalize the similarity score
    return sum(similarity_scores) / total_weight

# Calculate similarity for each row
similarity_scores = [
    compute_similarity(row, reference_dict, weights, ollama)
    for _, row in df.iterrows()
]

# Get the ordered list of row indices based on similarity scores
ordered_indices = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)

# Output the ordered list of row indices
print("Ordered list of row indices:", ordered_indices)