"""
Comprehensive benchmark and evaluation runner on 10 real memes from the dataset.
Compares Cultural-Aware mode vs General mode on:
- Latency (OCR, Preproc, Generation, Total)
- Prediction & Dual Probabilities P(H) and P(NH)
- Cultural Category & Dependency & Context
- Clean AI Reasoning (no raw JSON)
"""

import os
import sys
import time
import json
import pandas as pd
from PIL import Image

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.vlm.model_loader import load_model_and_processor
from src.vlm.inference import analyze_meme
from src.cultural.ocr_engine import extract_text_ocr
from src.cultural.context_retriever import retrieve_cultural_context

# 10 curated diverse real memes from data/processed/test.csv
BENCHMARK_MEMES = [
    {
        "filename": "train_1147.jpg",
        "description": "Indian Mom / Family - Kids returning late panic",
        "expected_theme": "family_relations / Indian Mom"
    },
    {
        "filename": "train_968.jpg",
        "description": "Indian TV Serials - CID, Diya Aur Baati, Taarak Mehta",
        "expected_theme": "bollywood_pop / Indian TV"
    },
    {
        "filename": "train_1990.jpg",
        "description": "Internet Viral - Binod spam meme",
        "expected_theme": "internet_culture / Binod viral trend"
    },
    {
        "filename": "train_2573.jpg",
        "description": "News / Journalism - Undercover journalists (Non-humorous)",
        "expected_theme": "politics_news / serious news"
    },
    {
        "filename": "train_2829.jpg",
        "description": "Bollywood Dialogue - 'Dhokhaa swabhaav hai mera'",
        "expected_theme": "bollywood_pop / Hindi cinema dialogues"
    },
    {
        "filename": "train_2788.jpg",
        "description": "Indian Education - Daily routine of a Kota student",
        "expected_theme": "education_exams / Kota coaching life"
    },
    {
        "filename": "train_3062.jpg",
        "description": "Desi Household - Father fixes non-working TV by hitting it",
        "expected_theme": "desi_lifestyle / household habits"
    },
    {
        "filename": "train_481.jpg",
        "description": "Cricket & Politics - Rahul Dravid election voting irony",
        "expected_theme": "cricket_sports / Indian elections"
    },
    {
        "filename": "train_5050.jpg",
        "description": "YouTube / Hinglish - 'Death threats bhi deta hu' roasting",
        "expected_theme": "internet_culture / YouTube roaster Hinglish"
    },
    {
        "filename": "train_6759.jpg",
        "description": "Desi Diet - Trying to diet but parents buy junk food",
        "expected_theme": "family_relations / desi parenting"
    }
]

def run_benchmark():
    images_dir = os.path.join(PROJECT_ROOT, "data", "processed", "images")
    test_csv_path = os.path.join(PROJECT_ROOT, "data", "processed", "test.csv")
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Load test metadata to get gold labels
    df_test = pd.read_csv(test_csv_path)
    metadata_map = {}
    for _, row in df_test.iterrows():
        fname = str(row.get("image_filename", ""))
        if fname:
            metadata_map[fname] = {
                "gold_humour": str(row.get("humour", "unknown")),
                "gold_is_humorous": int(row.get("is_humorous", 1)) if pd.notnull(row.get("is_humorous")) else 1,
                "dataset_ocr": str(row.get("ocr", ""))
            }

    print("=" * 80)
    print("STARTING 10-MEME BENCHMARK EVALUATION (CULTURAL-AWARE VS GENERAL)")
    print("=" * 80)

    # Load model
    print("\n[Step 1/3] Loading Qwen2.5-VL-3B-Instruct model...")
    t_load_start = time.perf_counter()
    model, processor, device = load_model_and_processor(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        min_pixels=200704,
        max_pixels=401408
    )
    load_time = time.perf_counter() - t_load_start
    print(f"Model loaded successfully in {load_time:.2f}s on {device}.")

    results = []

    print("\n[Step 2/3] Running evaluation on 10 memes...")
    for idx, item in enumerate(BENCHMARK_MEMES, 1):
        fname = item["filename"]
        img_path = os.path.join(images_dir, fname)
        gold = metadata_map.get(fname, {"gold_humour": "unknown", "gold_is_humorous": 1, "dataset_ocr": ""})

        print(f"\n[{idx}/10] Processing meme: {fname} ({item['description']})")
        if not os.path.exists(img_path):
            print(f"  WARNING: Image not found at {img_path}, skipping.")
            continue

        # Step A: OCR Extraction
        t_ocr_0 = time.perf_counter()
        ocr_result = extract_text_ocr(img_path)
        detected_text = ocr_result[0] if isinstance(ocr_result, (tuple, list)) else str(ocr_result)
        t_ocr = time.perf_counter() - t_ocr_0
        print(f"  OCR extracted ({t_ocr:.2f}s): {repr(detected_text[:60])}")

        # Step B: Cultural Context Retrieval
        t_ret_0 = time.perf_counter()
        cultural_ctx = retrieve_cultural_context(detected_text) if detected_text else ""
        t_ret = time.perf_counter() - t_ret_0

        # Step C: Cultural-Aware Inference (Mode A)
        t_ca_0 = time.perf_counter()
        res_ca = analyze_meme(
            model=model,
            processor=processor,
            image_input=img_path,
            include_context=True,
            cultural_context_override=cultural_ctx
        )
        t_ca = time.perf_counter() - t_ca_0
        lat_ca = res_ca.get("latency_seconds", t_ca)

        # Step D: General Inference (Mode B - no cultural context injected)
        t_gen_0 = time.perf_counter()
        res_gen = analyze_meme(
            model=model,
            processor=processor,
            image_input=img_path,
            include_context=False
        )
        t_gen = time.perf_counter() - t_gen_0
        lat_gen = res_gen.get("latency_seconds", t_gen)

        # AI reasoning verification (ensure no raw JSON braces)
        reasoning_clean_ca = not (res_ca.get("reasoning", "").strip().startswith("{") and res_ca.get("reasoning", "").strip().endswith("}"))
        reasoning_clean_gen = not (res_gen.get("reasoning", "").strip().startswith("{") and res_gen.get("reasoning", "").strip().endswith("}"))

        record = {
            "index": idx,
            "filename": fname,
            "description": item["description"],
            "expected_theme": item["expected_theme"],
            "gold_label": gold["gold_humour"],
            "gold_is_humorous": gold["gold_is_humorous"],
            "detected_ocr_text": detected_text,
            "ocr_latency_sec": round(t_ocr, 2),
            "cultural_aware": {
                "prediction": res_ca.get("prediction", "Unknown"),
                "p_humorous": res_ca.get("confidence", 0.5),
                "p_not_humorous": res_ca.get("p_not_humorous", 0.5),
                "cultural_category": res_ca.get("cultural_category", "none"),
                "cultural_dependency": res_ca.get("cultural_dependency", "none"),
                "cultural_context": res_ca.get("cultural_context", ""),
                "reasoning": res_ca.get("reasoning", ""),
                "reasoning_is_clean": reasoning_clean_ca,
                "latency_sec": round(lat_ca, 2)
            },
            "general_baseline": {
                "prediction": res_gen.get("prediction", "Unknown"),
                "p_humorous": res_gen.get("confidence", 0.5),
                "p_not_humorous": res_gen.get("p_not_humorous", 0.5),
                "reasoning": res_gen.get("reasoning", ""),
                "reasoning_is_clean": reasoning_clean_gen,
                "latency_sec": round(lat_gen, 2)
            }
        }
        results.append(record)

        print(f"  [CA Mode] Pred: {res_ca.get('prediction')} | P(H): {res_ca.get('confidence'):.2f} | Category: {res_ca.get('cultural_category')} | Latency: {lat_ca:.2f}s")
        print(f"  [Gen Mode] Pred: {res_gen.get('prediction')} | P(H): {res_gen.get('confidence'):.2f} | Latency: {lat_gen:.2f}s")

    print("\n[Step 3/3] Saving results and generating comparison summary...")
    
    # Save JSON
    json_path = os.path.join(results_dir, "benchmark_10_memes.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved detailed JSON results to {json_path}")

    # Flatten and save CSV
    csv_rows = []
    for r in results:
        csv_rows.append({
            "Meme": r["filename"],
            "Gold Label": r["gold_label"],
            "CA Prediction": r["cultural_aware"]["prediction"],
            "CA P(Humor)": r["cultural_aware"]["p_humorous"],
            "CA P(Not Humor)": r["cultural_aware"]["p_not_humorous"],
            "CA Category": r["cultural_aware"]["cultural_category"],
            "CA Dependency": r["cultural_aware"]["cultural_dependency"],
            "CA Latency (s)": r["cultural_aware"]["latency_sec"],
            "Gen Prediction": r["general_baseline"]["prediction"],
            "Gen P(Humor)": r["general_baseline"]["p_humorous"],
            "Gen Latency (s)": r["general_baseline"]["latency_sec"],
            "OCR Latency (s)": r["ocr_latency_sec"]
        })
    df_res = pd.DataFrame(csv_rows)
    csv_path = os.path.join(results_dir, "benchmark_10_memes.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"Saved summary CSV to {csv_path}")

    # Latency Stats
    ca_latencies = [r["cultural_aware"]["latency_sec"] for r in results]
    gen_latencies = [r["general_baseline"]["latency_sec"] for r in results]
    avg_ca_lat = sum(ca_latencies) / len(ca_latencies) if ca_latencies else 0
    avg_gen_lat = sum(gen_latencies) / len(gen_latencies) if gen_latencies else 0

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Total Memes Tested: {len(results)}")
    print(f"Average Latency (Cultural-Aware): {avg_ca_lat:.2f}s per meme")
    print(f"Average Latency (General Baseline): {avg_gen_lat:.2f}s per meme")
    print("=" * 80)
    print(df_res.to_string(index=False))
    print("=" * 80)

if __name__ == "__main__":
    run_benchmark()
