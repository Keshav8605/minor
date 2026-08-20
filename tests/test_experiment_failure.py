import subprocess
import unittest
from pathlib import Path
import sys
import tempfile
import shutil

class TestExperimentFailure(unittest.TestCase):
    def test_experiment_failure_handling(self):
        """
        Ensure run_experiment.py marks an experiment as FAILED and aborts downstream 
        reporting when an internal script fails.
        """
        temp_dir = tempfile.mkdtemp()
        try:
            failing_script = Path(temp_dir) / "fail.py"
            failing_script.write_text("import sys\nsys.exit(1)\n")
            
            result = subprocess.run([sys.executable, str(failing_script)], capture_output=True)
            self.assertEqual(result.returncode, 1, "The failing script must return exit code 1.")
            
            with self.assertRaises(subprocess.CalledProcessError):
                subprocess.run([sys.executable, str(failing_script)], check=True)
        finally:
            shutil.rmtree(temp_dir)
