import numpy as np
from typing import Dict, List
from tvm import autotvm
from tvm import relay
from collections import OrderedDict, defaultdict
from tvm.autotvm.task.space import SplitEntity, ConfigEntity, ReorderEntity, OtherOptionEntity, AnnotateEntity
import statistics
import tvm
import argparse

from visualizer import violin_plot, sensitivity_plot


import custom

parser = argparse.ArgumentParser()
parser.add_argument('log_file', type=str, default='')
parser.add_argument('-n_sample', type=int, default=-1)

args = parser.parse_args()


def analyze_running_data(log_file):
    task = None
    profiles = []
    for i, record in enumerate(autotvm.record.load_from_file(log_file)):
        if args.n_sample != -1 and len(profiles) >= args.n_sample:
            break
        measure_input, measure_result = record
        assert isinstance(measure_input, autotvm.measure.MeasureInput)
        assert isinstance(measure_result, autotvm.measure.MeasureResult)
        if measure_result.error_no != 0:
            continue
        target, task, config = measure_input
        # selected_results = sorted(measure_result.costs)[5:5+15]
        costs = statistics.mean(measure_result.costs)
        config = config._entity_map
        cfg = OrderedDict()
        for name, entity in config.items():
            if isinstance(entity, SplitEntity):
                cfg[name] = entity.size
            elif isinstance(entity, ConfigEntity):
                cfg[name] = entity.index
            elif isinstance(entity, ReorderEntity):
                cfg[name] = entity.perm
            elif isinstance(entity, OtherOptionEntity):
                cfg[name] = entity.val
            elif isinstance(entity, AnnotateEntity):
                cfg[name] = entity.anns
            else:
                raise ValueError("")
        profiles.append([cfg, costs])
    return task, profiles


def tuplize(x):
    if isinstance(x, (tuple, list)):
        return tuple(tuplize(v) for v in x)
    else:
        return x


def sensitivity(profiles):
    # subspace names and items
    subspace_names = list(profiles[0][0].keys())

    # schedule to latency mapping
    sch2latency = {}
    for profile in profiles:
        cfg, latency = profile
        sch2latency[tuple(tuplize(v) for v in cfg.values())] = latency
    latencies = [l for _, l in profiles]
    min_latency = min(latencies)

    # classifiy profiles into collections
    n = len(subspace_names)
    collections: Dict[int, Dict[tuple, List[float]]] = defaultdict(lambda:defaultdict(list))
    for sch, latency in sch2latency.items():
        if latency < 0.0:
            continue
        for i in range(n):
            msch = sch[0:i] + (None,) + sch[i+1:]
            collections[i][msch].append(latency)

    # calculate the sensitivity
    sensitivities = []
    for i in range(n):
        varanences = []
        for msch, latencies in collections[i].items():
            varanences.append(statistics.pstdev(latencies) / min_latency)
            # varanences.append((max(latencies) - min(latencies)) / min_latency)
            # varanences.append((max(latencies) - min(latencies)) / min_latency)
        sensitivities.append(statistics.mean(varanences))
    return {name: s for name, s in zip(subspace_names, sensitivities)}


def schedule_latency_lines(profiles):  # profiles: List[Tuple[Dict[name, value], Latency]]
    subspace_names = list(profiles[0][0].keys())
    columns = []
    for name in subspace_names:
        value = profiles[0][0][name]
        if isinstance(value, (list, tuple)):
            assert value[0] == -1 and all(v != -1 for v in value[1:])
            for i in range(1, len(value)):
                columns.append(f'{name}_{i}')
        else:
            columns.append(name)
    columns.append('latency')

    lines = []
    for subspaces, latency in profiles:
        line = []
        for name in subspace_names:
            value = subspaces[name]
            if isinstance(value, (tuple, list)):
                line.extend(value[1:])
            else:
                line.append(subspaces[name])
        line.append(latency)
        lines.append(line)
    return columns, lines


def main():
    log_file: str = args.log_file
    task, profiles = analyze_running_data(log_file)
    assert isinstance(task, autotvm.task.Task)

    prefix = log_file
    if prefix.endswith('.log'):
        prefix = prefix[:-4]

    with open(f'{prefix}.summary', 'w') as f:
        n_samples = len(profiles)
        summary = [
            f'Task name: {task.name}',
            f'Args: {task.args}',
            f'Num of samples: {n_samples}',
            f'Log file: {log_file}',
            '\n'
        ]
        f.write("\n".join(summary))

    with open(f'{prefix}.sensitivity', 'w') as f:
        sensitivities = sensitivity(profiles)
        name_length = max(len(name) for name in sensitivities.keys())
        for name, value in sensitivities.items():
            f.write(f'{name:>{name_length}}: {value:.3f}\n')

    with open(f'{prefix}.table', 'w') as f:
        head, lines = schedule_latency_lines(profiles)
        f.write(' '.join(head) + '\n')
        for line in lines:
            f.write(' '.join(str(v) for v in line) + '\n')

    sensitivity_plot(sensitivities, fname=f'{prefix}.sensitivity.pdf')
    violin_plot(profiles, fname=f'{prefix}.violin.pdf')



if __name__ == '__main__':
    main()
