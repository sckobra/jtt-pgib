import re
import matplotlib.pyplot as plt


def plot_metric(log_file, metric, split='test'):
    """
    Parse a training log and plot a metric vs. epoch.

    Args:
        log_file: path to the .txt log file
        metric:   metric name, e.g. 'acc', 'loss', 'auc', 'fid+', 'charact', 'worst_group_acc'
        split:    which set to pull values from — 'train', 'eval', or 'test'
    """
    prefix_map = {
        'train': r'^Epoch\s+(\d+)',
        'eval':  r'^Eval epoch\s+(\d+)',
        'test':  r'^Test epoch\s+(\d+)',
    }
    if split not in prefix_map:
        raise ValueError(f"split must be one of {list(prefix_map)}, got '{split}'")

    prefix_re = re.compile(prefix_map[split])
    # matches   key: value   where value is a plain float
    metric_re = re.compile(rf'(?<![:\w]){re.escape(metric)}:\s*([\d.]+)')

    epochs, values = [], []

    with open(log_file) as f:
        for line in f:
            m = prefix_re.match(line)
            if not m:
                continue
            epoch = int(m.group(1))
            v = metric_re.search(line)
            if v is None:
                raise ValueError(f"Metric '{metric}' not found in {split} lines. "
                                 f"Check the metric name or try a different split.")
            epochs.append(epoch)
            values.append(float(v.group(1)))

    if not epochs:
        raise ValueError(f"No {split} lines found in {log_file}")

    plt.figure()
    plt.plot(epochs, values, marker='o', markersize=3, linewidth=1.5)
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.title(f'{split.capitalize()} {metric} vs. Epoch')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
