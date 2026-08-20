# Cultural-Awareness Module

## What does Cultural Awareness mean?
In the context of this project, cultural awareness refers to the system's ability to explicitly retrieve and supply relevant factual contexts about Indian society (e.g., family dynamics, education systems, Bollywood) to a Vision-Language Model (VLM). 

## Why it matters for Hindi/Hinglish Humor
Codemixed memes often rely on deeply ingrained cultural tropes. A base VLM might translate "Sharma ji ka beta" literally to "Mr. Sharma's son", missing the metaphorical implication of the overachieving peer that Indian parents compare their children to. Without this context, humor detection frequently fails.

## The Taxonomy
We use a controlled JSON taxonomy of 17 categories (e.g., `family`, `JEE/exams`, `cricket`). These were chosen because they frequently appear in Memotion 3 codemixed data.

## How Retrieval Works
1. **Keyword/Heuristic Detection**: Basic OCR text or initial visual descriptions are scanned against a dictionary of cultural keywords (e.g., "kota", "neet" maps to `JEE/exams`).
2. **Context Fetching**: If a category is detected, a concise factual description is fetched from `cultural_knowledge.json`.
3. **Context Injection**: The retrieved facts are injected into the VLM prompt under `EXTERNAL CULTURAL CONTEXT`. 

## Limitations & "Not Full Understanding"
This system is a lightweight, static heuristic framework. It does *not* grant the VLM an intuitive, intrinsic understanding of Indian culture. It merely acts as a factual dictionary. It cannot catch subtle visual tropes or highly contextual slang that doesn't trigger the explicit keyword mapping. It is a baseline RAG (Retrieval-Augmented Generation) substitute to prove the hypothesis that external cultural context improves zero-shot reasoning.
