"""Independent T3 scoring; see --help for required paths."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoring.task_entry import cli, main as score_task


def main(argv=None):
    return score_task('T3', argv)


if __name__ == '__main__':
    cli('T3')
