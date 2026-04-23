import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from Configures import DataParser


def save_run_log():
    data_args = DataParser()
    dataset_name = data_args.dataset_name

    log_dir = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(log_dir, 'hyper_search.txt')

    if not os.path.isfile(src):
        print(f"No hyper_search.txt found at {src}")
        return

    timestamp = datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
    dst = os.path.join(log_dir, f'results_{dataset_name}_{timestamp}.txt')

    with open(src, 'r') as f:
        contents = f.read()
    with open(dst, 'w') as f:
        f.write(contents)

    print(f"Saved to {dst}")


if __name__ == '__main__':
    save_run_log()
