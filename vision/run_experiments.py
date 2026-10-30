"""Run experiments sequentially, saving stdout, time -v and one-second resource samples."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    cfg = json.loads((ROOT / 'configs/vision_experiment.json').read_text())
    output = ROOT / cfg['output']
    output.mkdir(parents=True, exist_ok=True)
    for model in cfg['models']:
        time_file = output / f'{model}_time.txt'
        if (output / model).exists():
            if time_file.exists() and 'Exit status: 0' in time_file.read_text():
                print(f'Skipping completed {model}', flush=True)
                continue
            raise SystemExit(f'Existing unfinished output for {model}; move it aside before retrying.')
        print(f'Starting {model}', flush=True)
        with (output / f'{model}.log').open('w') as log, (output / f'{model}_resources.csv').open('w') as samples:
            writer = csv.writer(samples)
            writer.writerow(['elapsed_seconds', 'process_tree_rss_bytes', 'system_cpu_percent', 'system_available_memory_bytes'])
            start = time.monotonic()
            process = subprocess.Popen(['/usr/bin/time', '-v', '-o', str(output / f'{model}_time.txt'),
                sys.executable, '-u', str(ROOT / 'vision/train_detector.py'), model], stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
            parent = psutil.Process(process.pid)
            while process.poll() is None:
                rss = 0
                try:
                    for child in [parent, *parent.children(recursive=True)]:
                        try:
                            rss += child.memory_info().rss
                        except psutil.Error:
                            pass
                except psutil.Error:
                    pass
                writer.writerow([round(time.monotonic()-start, 2), rss, psutil.cpu_percent(), psutil.virtual_memory().available])
                samples.flush()
                time.sleep(1)
            print(f'{model}: exit={process.returncode}, elapsed={time.monotonic()-start:.1f}s', flush=True)
            if process.returncode:
                raise SystemExit(f'{model} failed. See {output / (model + ".log")}')
