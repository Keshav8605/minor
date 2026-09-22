"""
Verification script for the RAG pipeline in Hindi-Humor-VLM.
Tests CulturalRetriever semantic retrieval with Query A (Match) and Query B (No Match).
"""

import sys
from pathlib import Path

# Ensure root directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.cultural.context_retriever import CulturalRetriever


def run_verification():
    print("=== RAG Pipeline Verification Script ===")
    print("Initializing CulturalRetriever...")
    retriever = CulturalRetriever()

    query_a = "sharma ji ka beta"
    query_b = "random meaningless text with no cultural context"

    print(f"\n--- QUERY A (Should Match): '{query_a}' ---")
    result_a = retriever.retrieve_context(query_a)
    print("RESULT_A_START")
    print(result_a)
    print("RESULT_A_END")

    print(f"\n--- QUERY B (No Match): '{query_b}' ---")
    result_b = retriever.retrieve_context(query_b)
    print("RESULT_B_START")
    print(result_b)
    print("RESULT_B_END")


if __name__ == "__main__":
    run_verification()
