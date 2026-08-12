topic = 'Semantic Bug Matching & Solution Lookup'

import os
from dotenv import load_dotenv

#read perms only
load_dotenv()

#imported AFTER HT_TOKEN is loaded
from sentence_transformers import SentenceTransformer
from typing import Optional, Any
import numpy as np 

#model is trained for symmetric sentence similarity
model = SentenceTransformer(model_name_or_path='all-MiniLM-L6-v2')


#workflow -> semantic query search, if not exact case found, query bug to llm api
#llm api response added as fix for respective bug_signatures
bug_database = [
    {
    'bug_signature': 'IndexError: list index out of range in binary search', 
    'code_context': 'while low <= high: mid = (low + high) / 2; if arr[mid] == target:',
    'fix': 'Use integer division "//" instead of "/" to prevent float indices: mid = (low + high) // 2' 
    },
    {
    'bug_signature': 'RecursionError: maximum recursion depth exceeded in fibonacci',
    'code_context': 'def fib(n): return fib(n-1) + fib(n-2)',
    'fix': 'Add a base case to stop recursion: if n <= 1: return n'
    },
    {
    'bug_signature': 'AttributeError: "NoneType" object has no attribute "val" in linked list', 
    'code_context': 'while curr.next.val != target: curr = curr.next',
    'fix': 'Check if curr or curr.next is None before accessing .val: while curr and curr.next:'
    }
]

bug_signatures = [f'{bug['bug_signature']} | Context: {bug['code_context']}' for bug in bug_database]
database_embeddings = model.encode(bug_signatures, convert_to_numpy=True)

input_error = 'IndexError: list index out of range'
input_code = 'mid = (start + end) / 2 \n print(my_list[mid])'

query_text = f'{input_error} | Context: {input_code}'
query_embeddings = model.encode(query_text, convert_to_numpy=True)

#model.similarity handles n x n matrix multiply ops, where n => 1
similarity_scores = model.similarity(database_embeddings, query_embeddings)
scores = similarity_scores.numpy().flatten()
best_match_idx = np.argmax(scores)

print(f'Closest Historical Bug Match (Confidence: {scores[best_match_idx]:.2f})')
print(f'-> Verified Fox: {bug_database[best_match_idx]['fix']}')