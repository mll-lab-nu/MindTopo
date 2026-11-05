"""Count the fixed release after strict image preflight, without silent filtering."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "topobench_eval"))
from topobench_eval.datasets import question_path, validate_dataset

def _question_jsonl_path(task_name):
    return question_path(task_name)

def get_or_compute(task_name):
    return len(validate_dataset(task_name))

if __name__ == "__main__":
    print(get_or_compute(sys.argv[1]))
