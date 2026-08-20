# Vision-Language Model (VLM) Inference

This document explains the core concepts behind the VLM inference engine built in Phase 3.

## What is a VLM?
A Vision-Language Model (VLM) is an AI model that can simultaneously process both text and images. Unlike traditional language models (like ChatGPT without vision) that only understand text, or traditional vision models (like ResNet) that only classify images, a VLM can look at an image and answer questions about it, read the text inside it, and reason about its visual content.

## What is Qwen2.5-VL?
Qwen2.5-VL is an open-weights Vision-Language Model developed by Alibaba. The `Qwen2.5-VL-3B-Instruct` version we are using has 3 Billion parameters and is optimized for instruction-following. It is particularly good at reading text within images (OCR) and reasoning about complex visual inputs, which is critical for memes containing Hindi, English, and codemixed text.

## What does the Processor do?
In Hugging Face Transformers, a model only understands numbers (tensors). The `AutoProcessor` is responsible for converting our raw inputs—like a JPEG image file and a string of text—into the exact numerical tensors that the model expects. It handles tokenizing the text and resizing/normalizing the images.

## What does Inference mean?
"Inference" is the phase where we use an already trained model to make predictions. In our case, inference means passing a meme image and our prompt into the model and generating an analysis. We are not training or fine-tuning the model in this phase.

## What does the Prompt do?
The prompt is the set of instructions we give the model. We use a **System Prompt** to define the model's persona (e.g., "You are an expert AI assistant..."). We use a **User Prompt** to pass the image and ask specific questions (e.g., "Analyze the following meme...").

## What is Structured Output?
By default, VLMs generate free-form text (prose). However, for a data pipeline, we need predictable, machine-readable output. We instruct the model to return a strict JSON format (Structured Output) so that our code can reliably parse the answer into variables like `humorous` (boolean), `confidence` (float), and `reason` (string). Our output parser includes a fallback mechanism: if the model fails to output valid JSON, it captures the raw text rather than fabricating false data.
