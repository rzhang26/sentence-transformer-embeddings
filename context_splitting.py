topic = 'Contextual AI Tutoring (RAG Prompt Prep'

import os 
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder, util


HF_TOKEN = os.getenv('HF_TOKEN', None)

#model is trained for assymetric similarity
bi_encoder = SentenceTransformer(model_name_or_path='BAAI/bge-small-en-v1.5', use_auth_token=HF_TOKEN)
cross_encoder = CrossEncoder(model_name_or_path='cross-encoder/ms-marco-MiniLM-L-6-v2', token=HF_TOKEN)

gemini_analysis_chunks = [
    'Line 4 triggers an O(N) linear time operation because it iterates completely through the unsorted array.',
    'The auxiliary space complexity scales to O(N) on Line 8 because a new copy of the array is allocated in memory.',
    'Line 12 represents the base case of the recursion stack where execution safely returns if the node pointer is null.',
    'The worst-case runtime remains O(N^2) because of the nested loops operating on lines 5 and 7 respectively.'
]

context_embeddings = bi_encoder.encode(
    gemini_analysis_chunks,
    normalize_embeddings=True, #helps smooth matrix multiply process
    convert_to_tensor=True
)

#sample inputs
student_question = 'Why did my memory usage go up during the middle of the run?'
clicked_line = 8

#custom algorithm for calculating similarity scores btwn Qs & As
# - utilizes the Retrieve-and-Rerank architecture (RaR)
# - retrieves most likely candidates (~top K) via cos-similarity through bi-encoder 
# - rerank candidates from new subset to accurately find best matches via cross-encoder
# --> semantic relevance processing ([key, doc] pairs)
# --> line reference searching (+0.5 to canddiate score)
# - returns candidate(s) as output
question_embedding = bi_encoder.encode(
    f'Represent this sentence for searching relevant passages: {student_question}',
    normalize_embeddings=True, #helps smooth matrix multiply process
    convert_to_tensor=True
)

raw_cos_sim = util.cos_sim(question_embedding, context_embeddings)[0].cpu().numpy()

TOP_K = 2
top_k_indices = np.argsort(raw_cos_sim)[::-1][:TOP_K]

candidate_chunks = [[student_question, gemini_analysis_chunks[idx]] for idx in top_k_indices]
cross_scores = cross_encoder.predict(candidate_chunks)

final_candidate_scores = np.copy(cross_scores)
for i, chunk in enumerate(candidate_chunks):
    if f'Line {clicked_line}' in chunk:
        final_candidate_scores[i] += 0.5

best_candidate_pos = np.argmax(final_candidate_scores)
best_candidate_idx = top_k_indices[best_candidate_pos]

print(f"Top-{TOP_K} Bi-Encoder Candidates Indices: {top_k_indices}")
print(f"Student Query: '{student_question}'")
print(f"Extracted/Curated Context for LLM:\n-> '{gemini_analysis_chunks[best_candidate_idx]}'")