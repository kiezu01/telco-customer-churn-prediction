import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_step(script_name: str) -> None:
    print(f"\n=== Running {script_name} ===")
    subprocess.run([sys.executable, os.path.join(BASE_DIR, 'src', script_name)], check=True)


if __name__ == '__main__':
    run_step('preprocess.py')
    run_step('model.py')
    run_step('explain.py')
    print('\nDemo completed successfully!')
