import yaml
import datetime
import uuid
from pathlib import Path
import json

class ExperimentRegistry:
    def __init__(self, registry_path: str = "experiments/registry.yaml"):
        with open(registry_path, "r") as f:
            self.registry = yaml.safe_load(f)["experiments"]
            
    def initialize_run(self, exp_id: str):
        if exp_id not in self.registry:
            raise ValueError(f"Experiment {exp_id} not found in registry.")
            
        exp_meta = self.registry[exp_id]
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        run_uuid = str(uuid.uuid4())[:8]
        run_id = f"{exp_id}_{timestamp}_{run_uuid}"
        
        base_dir = Path("results")
        run_dir = base_dir / exp_id / run_id
        
        for subdir in ["reports", "tables", "figures", "raw_predictions", "models"]:
            (run_dir / subdir).mkdir(parents=True, exist_ok=True)
            
        record = {
            "run_id": run_id,
            "experiment_id": exp_id,
            "date": datetime.datetime.now().isoformat(),
            "name": exp_meta["name"],
            "type": exp_meta["type"],
            "base_config": exp_meta["model_config"],
            "cultural_mode": exp_meta.get("cultural_mode", None),
            "run_dir": str(run_dir)
        }
        
        record["status"] = "RUNNING"
        with open(run_dir / "run_metadata.json", "w") as f:
            json.dump(record, f, indent=2)
            
        return record
        
    def update_status(self, run_dir: str, status: str):
        meta_path = Path(run_dir) / "run_metadata.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                record = json.load(f)
            record["status"] = status
            with open(meta_path, "w") as f:
                json.dump(record, f, indent=2)
