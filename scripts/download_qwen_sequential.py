"""Download Qwen2.5-VL-3B-Instruct model weights sequentially with clear logging."""
import sys
import time
from huggingface_hub import hf_hub_download

repo_id = "Qwen/Qwen2.5-VL-3B-Instruct"
files = [
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors"
]

for filename in files:
    print(f"[{time.strftime('%H:%M:%S')}] Downloading {filename}...", flush=True)
    t0 = time.time()
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
        )
        elapsed = time.time() - t0
        print(f"[{time.strftime('%H:%M:%S')}] Finished {filename} in {elapsed:.1f}s -> {path}", flush=True)
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] Error downloading {filename}: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

print(f"[{time.strftime('%H:%M:%S')}] All weights successfully downloaded!", flush=True)
