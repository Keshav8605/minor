import json
import os
from src.cultural.context_builder import CulturalContextBuilder

builder = CulturalContextBuilder(
    categories_path="data/cultural/cultural_categories.json",
    knowledge_path="data/cultural/cultural_knowledge.json",
    drishtikon_path="data/cultural/drishtikon/processed/drishtikon_knowledge.json",
    drishtikon_enabled=True,
    drishtikon_top_k=3,
    min_similarity_threshold=0.15
)

test_queries = [
    ("Indian family", "mummy papa rishtedar joint family wedding sanskar"),
    ("Education / JEE / College", "engineering college hostel semester exam backlog topper coaching kota"),
    ("Cricket / Sports", "dhoni virat kohli ipl cricket match world cup stadium wicket"),
    ("Bollywood / Media", "bollywood movie actor dialogue hera pheri cinema baburao"),
    ("Festivals", "diwali puja rangoli holi festival celebration crackers sweets"),
    ("Religion", "mandir temple ganga arti puja kedarnath prasad holy shrine"),
    ("Food / Cuisine", "samosa chai pani puri biryani chole bhature street food"),
    ("Regional Culture", "punjabi bhangra garba gujarat bihar south indian dosa local train"),
    ("Hinglish References", "log kya kahenge jugaad sharma ji ka beta nibba nibbi")
]

results = {}
for name, q in test_queries:
    ctx, sources, records = builder.build_context(q)
    results[name] = {
        "query": q,
        "sources": sources,
        "matches_count": len(records),
        "top_match": records[0] if records else None,
        "context_snippet": ctx[:250] if ctx else ""
    }
    print(f"=== {name} ===")
    print(f"  Sources: {sources}")
    print(f"  Matches: {len(records)}")
    if records:
        top = records[0]
        print(f"  Top score: {top.get('relevance_score')}")
        print(f"  Attribute: {top.get('attribute')} | State: {top.get('state')} | Lang: {top.get('language')}")
        print(f"  Question: {top.get('question')}")
        print(f"  Answer: {top.get('answer')}")
    print()

os.makedirs("results/evaluation", exist_ok=True)
with open("results/evaluation/drishtikon_retrieval_validation.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("Saved validation results to results/evaluation/drishtikon_retrieval_validation.json")
