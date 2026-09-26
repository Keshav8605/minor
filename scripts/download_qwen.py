"""Script to download and verify Qwen2.5-VL-3B-Instruct model weights."""
import sys
import time
from huggingface_hub import snapshot_download

def main():
    repo_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    print(f"Starting download of {repo_id}...")
    t0 = time.time()
    try:
        path = snapshot_download(
            repo_id=repo_id,
            resume_download=True,
            max_workers=4
        )
        print(f"Successfully downloaded {repo_id} to {path} in {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
