from collections import OrderedDict, defaultdict
from typing import Dict, List
import tvm
import statistics
from tqdm import tqdm
from tvm import autotvm
import main_tvm_gpu_conv
from operator import mul
from functools import reduce



def analyze_task():
    print("analyze_task")
    for record in autotvm.record.load_from_file('conv2d_grid_search.log'):
        measure_input, _= record
        target, task, _= measure_input
        task = autotvm.task.create(task.name, args=task.args, target=target)
        products = OrderedDict()
        for name, subspace in task.config_space.space_map.items():
            products[name] = len(subspace)
        # manual define the depths currently. Add automatic analyze in the future
        # depths = OrderedDict()
        # depths['tile_f'] = [0, 3, 6, 9]
        # depths['tile_y'] = [1, 4, 7, 10]
        # depths['tile_x'] = [2, 5, 8, 11]
        # depths['tile_rc'] = [12, 15, 18]
        # depths['tile_ry'] = [13, 16, 19]
        # depths['tile_rx'] = [14, 17, 20]
        # return depths, products
        return products

def load_profiles(products):
    print("load_profiles")
    profiles = []
    
    for i, record in enumerate(autotvm.record.load_from_file('conv2d_grid_search.log')):
        measure_input, measure_result = record
        target, task, config = measure_input
        selected_results = sorted(measure_result.costs)[5:5+15]
        if len(selected_results) == 0:
            costs = -1
        else:
            costs = sum(selected_results)/len(selected_results)
        config = config._entity_map
        cfg = OrderedDict()
        for name, split_entity in config.items():
            assert isinstance(split_entity, autotvm.task.space.SplitEntity)
            extents = split_entity.size
            cfg[name] = [products[name] // reduce(mul, extents[1:])] + extents[1:]
        profiles.append([cfg, costs])
    return profiles

def sensitivity(profiles):
    # subspace names and items
    subspace_names = list(profiles[0][0].keys())
    subspace_items = [set() for _ in range(len(subspace_names))]
    for profile in profiles:
        cfg, _ = profile
        for idx, vals in enumerate(cfg.values()):
            subspace_items[idx].add(tuple(vals))
    subspace_items = [list(st) for st in subspace_items]
    print(subspace_names)
    # for subspace in subspaces:
    #     print(subspace)
    
    # schedule to latency mapping
    sch2latency = {}
    for profile in profiles:
        cfg, latency = profile
        sch2latency[tuple(tuple(v) for v in cfg.values())] = latency

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
            varanences.append(statistics.pvariance(latencies))
        sensitivities.append(statistics.mean(varanences))
    print(sensitivities)


def main():
    # depths, products = analyze_task()
    products = analyze_task()
    print(products)
    profiles = load_profiles(products)
    print("hi")
    sensitivity(profiles)


if __name__ == '__main__':
    main()


