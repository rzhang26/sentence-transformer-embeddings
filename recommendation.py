topic = 'Next-Algorithm Recommendation Engine'

from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

# 1. Algorithm catalog with concept tags
algorithm_catalog = [
    {"name": "Quick Sort", "tags": "Divide and conquer array sorting algorithm with average O(N log N) time complexity."},
    {"name": "Binary Search Tree Insertion", "tags": "Pointer manipulation and structural tree traversal logic."},
    {"name": "Dijkstra's Shortest Path", "tags": "Graph traversal optimization using greedy tracking and priority queues."},
    {"name": "Merge Sort", "tags": "Divide and conquer stable sorting split and merge routine."},
    {"name": "Breadth-First Search", "tags": "Queue-driven layer by layer graph exploration layout."}
]

catalog_tags = [algo["tags"] for algo in algorithm_catalog]
catalog_embeddings = model.encode(catalog_tags, convert_to_numpy=True)

# 2. Simulate the user's active/completed visualizer history
user_just_finished = "Bubble Sort"
user_history_tags = "In-place iterative comparison sorting array manipulation with poor O(N^2) runtime performance."
user_embedding = model.encode(user_history_tags, convert_to_numpy=True)

# 3. Compute match scores across the available catalog
match_scores = np.dot(catalog_embeddings, user_embedding) / (
    np.linalg.norm(catalog_embeddings, axis=1) * np.linalg.norm(user_embedding)
)

# 4. Filter out any identical algorithmic matches and pick the next best conceptual step
ranked_indices = np.argsort(match_scores)[::-1]
print(f"Because you just visualized {user_just_finished}, you should try:")

for idx in ranked_indices:
    # Avoid recommending something that uses almost identical logic (like Merge Sort right after sorting)
    if "divide and conquer" in algorithm_catalog[idx]["tags"].lower():
        print(f"-> {algorithm_catalog[idx]['name']} (Recommendation match score: {match_scores[idx]:.2f})")
        break
