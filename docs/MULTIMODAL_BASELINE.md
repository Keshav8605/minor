# Classical Multimodal Baseline Architecture

## Why do we need this?
To scientifically prove that a Vision-Language Model (VLM) is better at detecting codemixed humor, we must compare it against the traditional, "classical" approach to multimodal machine learning.

## The Architecture
The conventional way to combine image and text is to train two separate, specialized models and combine their outputs into a single classifier.

1. **Vision Encoder (`google/vit-base-patch16-224`)**: A Vision Transformer. It chops an image into a grid of 16x16 patches and learns to extract visual features (like edges, shapes, faces). It outputs a 768-dimensional mathematical vector representing the image.
2. **Text Encoder (`xlm-roberta-base`)**: A massive multilingual language model. It excels at reading 100+ languages, including Hindi and English. It outputs a 768-dimensional vector representing the OCR text.
3. **Feature Fusion**: This is simply concatenation (Early Fusion). We stick the 768-dim image vector and the 768-dim text vector together to create a massive 1536-dimensional feature vector.
4. **MLP Classifier (Multi-Layer Perceptron)**: A standard neural network classifier with hidden layers (e.g., 256 dimensions) and dropout to prevent overfitting. It looks at the 1536-dim combined vector and outputs a single score predicting whether the meme is humorous or not.

## Modes Supported
- **Image-only**: Drops the text encoder.
- **Text-only**: Drops the image encoder.
- **Multimodal**: Uses both.

Comparing these modes allows us to measure whether the meme's humor strictly relies on text, image, or the synergy of both.
