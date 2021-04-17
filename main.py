import numpy as np
from typing import Dict, List
from tvm import autotvm
from tvm import topi, te, relay
from collections import OrderedDict, defaultdict
from tvm.autotvm.task.space import SplitEntity, ConfigEntity, ReorderEntity, OtherOptionEntity, AnnotateEntity
import statistics
from pprint import pprint
import tvm


def collect_running_data(task: autotvm.task.Task, tuner_name, log_file, n_trial=-1):
    n_total = len(task.config_space)
    if n_trial == -1:
        n_trial = n_total
    print(f'Task name: {task.name}')
    print(f'Args: {task.args}')
    print(f'Target: {task.target}')
    print(f'Num of trials: {n_trial} / {n_total} ({n_trial / n_total:.2f}%)')
    print(f'Log file: {log_file}')
    print('Schedule space:')
    print(task.config_space)

    # get tuner
    if tuner_name == 'grid_search':
        tuner = autotvm.tuner.GridSearchTuner(task)
    elif tuner_name == 'random':
        tuner = autotvm.tuner.RandomTuner(task)
    else:
        raise ValueError("")

    # start tuning
    tuner.tune(
        n_trial=n_trial,
        measure_option=autotvm.measure_option(
            builder=autotvm.LocalBuilder(),
            runner=autotvm.LocalRunner(
                number=1,
                repeat=25,
                timeout=4,
                min_repeat_ms=25
            )
            # runner=autotvm.RPCRunner(
            #     "local",
            #     "0.0.0.0",
            #     9190,
            #     number=1,
            #     repeat=25,
            #     timeout=4,
            #     min_repeat_ms=25
            # )
        ),
        callbacks=[
            autotvm.callback.log_to_file(log_file),
            autotvm.callback.progress_bar(n_trial)
        ]
    )


def analyze_running_data(log_file):
    profiles = []
    for i, record in enumerate(autotvm.record.load_from_file(log_file)):
        measure_input, measure_result = record
        target, task, config = measure_input
        selected_results = sorted(measure_result.costs)[5:5+15]
        if len(selected_results) == 0:
            costs = -1
        else:
            costs = sum(selected_results)/len(selected_results)
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
    return profiles


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
    min_latency = min(l for _, l in profiles)

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
        sensitivities.append(statistics.mean(varanences))
    return {name: s for name, s in zip(subspace_names, sensitivities)}


def workload():
    x = relay.var('input', shape=(1, 3, 224, 224))
    w = relay.var('w', shape=(64, 3, 3, 3))
    x = relay.nn.conv2d(x, w, padding=(1, 1))
    f = relay.Function(relay.analysis.free_vars(x), x)
    mod = tvm.IRModule.from_expr(f)
    mod = relay.transform.InferType()(mod)
    shape_dict = {v.name_hint: v.checked_type for v in mod["main"].params}
    params = {}
    for k, v in shape_dict.items():
        if k == "data":
            continue
        init_value = np.zeros(v.concrete_shape).astype(v.dtype)
        params[k] = tvm.nd.array(init_value, ctx=tvm.cpu(0))
    tasks = autotvm.task.extract_from_program(
        mod["main"], target='llvm', params=params, ops=(relay.op.get("nn.conv2d"),)
    )
    return tasks[0]


def main():
    task = workload()
    log_file = f'{task.name}.log'
    # collect_running_data(task, tuner_name='random', log_file=log_file, n_trial=100)
    profiles = analyze_running_data(log_file)
    # print(profiles)
    results = sensitivity(profiles)
    pprint(results)


if __name__ == '__main__':
    main()
