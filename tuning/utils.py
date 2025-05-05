from datetime import datetime
from pathlib import Path
def new_run_dir():
    run_dir = Path(__file__).parent / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
