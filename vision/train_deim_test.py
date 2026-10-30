"""Run the unmodified official DEIM trainer on CPU with four threads."""
from pathlib import Path
import runpy
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    torch.set_num_threads(4)
    vendor = ROOT / 'reference/DEIM'
    sys.path.insert(0, str(vendor))
    sys.argv = [str(vendor / 'train.py'), '-c', str(ROOT / 'data/sponge_training_test/deim.yml'),
                '-d', 'cpu', '--seed', '42', '-t', str(ROOT / 'models/deim_dfine_n_coco.pth')]
    runpy.run_path(str(vendor / 'train.py'), run_name='__main__')
