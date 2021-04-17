import numpy as np
from typing import Dict, List
from tvm import autotvm
from tvm import topi, te, relay
from collections import OrderedDict, defaultdict, namedtuple
from tvm.autotvm.task.space import SplitEntity, ConfigEntity, ReorderEntity, OtherOptionEntity, AnnotateEntity
import statistics
from pprint import pprint
import tvm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-command', type=str, choices=['collect', 'analyze'], default='analyze')
parser.add_argument('-task_type', type=str, choices=['conv2d', 'dense'], default='dense')
parser.add_argument('-args', type=str, default='1-8-8')
parser.add_argument('-target', type=str, choices=['llvm', 'cuda'], default='llvm')
parser.add_argument('-n_trial', type=int, default=1000)
parser.add_argument('-log_file', type=str, default='')

args = parser.parse_args()


def collect_running_data(task: autotvm.task.Task, tuner_name, log_file, n_trial=-1):
    n_total = len(task.config_space)
    if n_trial == -1:
        n_trial = n_total
    if n_trial > n_total:
        n_trial = n_total
    print(f'Task name: {task.name}')
    print(f'Args: {task.args}')
    print(f'Target: {task.target}')
    print(f'Num of trials: {n_trial} / {n_total} ({n_trial / n_total * 100:.2f}%)')
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
            varanences.append(statistics.pstdev(latencies) / min_latency * 100)
        sensitivities.append(statistics.mean(varanences))
    return {name: s for name, s in zip(subspace_names, sensitivities)}


def task_from_relay(x, target, op_name):
    f = relay.Function(relay.analysis.free_vars(x), x)
    mod = tvm.IRModule.from_expr(f)
    mod = relay.transform.InferType()(mod)
    shape_dict = {v.name_hint: v.checked_type for v in mod["main"].params}
    params = {}
    if target == 'llvm':
        ctx = tvm.cpu()
    elif target == 'cuda':
        ctx = tvm.gpu()
    else:
        raise ValueError("")
    for k, v in shape_dict.items():
        if k == "input":
            continue
        init_value = np.zeros(v.concrete_shape).astype(v.dtype)
        params[k] = tvm.nd.array(init_value, ctx=ctx)
    tasks = autotvm.task.extract_from_program(
        mod["main"], target=target, params=params, ops=(relay.op.get(op_name),)
    )
    return tasks[0]


def conv2d_task(batch_size, in_channels, h, w, out_channels, kernel, padding, strides, target):
    x = relay.var('input', shape=(batch_size, in_channels, h, w))
    w = relay.var('w', shape=(out_channels, in_channels, kernel[0], kernel[1]))
    x = relay.nn.conv2d(x, w, padding=padding, strides=strides)
    task = task_from_relay(x, target, "nn.conv2d")
    return task


def dense_task(batch_size, in_channels, out_channels, target):
    x = relay.var('input', shape=(batch_size, in_channels))
    w = relay.var('w', shape=(out_channels, in_channels))
    x = relay.nn.dense(x, w)
    task = task_from_relay(x, target, "nn.dense")
    return task


def collect_command():
    if args.task_type == 'conv2d':
        bs, ic, h, w, oc, kx, ky, px, py, sx, sy = [int(v) for v in args.args.split('-')]
        task = conv2d_task(batch_size=bs, in_channels=ic, h=h, w=w,
                           out_channels=oc, kernel=(kx, ky), padding=(px, py), strides=(sx, sy),
                           target=args.target)
    elif args.task_type == 'dense':
        bs, ic, oc = [int(v) for v in args.args.split('-')]
        task = dense_task(batch_size=bs, in_channels=ic, out_channels=oc, target=args.target)
    else:
        raise ValueError("")
    log_file = args.log_file if len(args.log_file) > 0 else f'{args.task_type}-{args.args}-{args.target}.log'
    collect_running_data(task, tuner_name='random', log_file=log_file, n_trial=args.n_trial)


def analyze_command():
    log_file = args.log_file
    profiles = analyze_running_data(log_file)
    results = sensitivity(profiles)
    pprint(results)


if __name__ == '__main__':
    if args.command == 'collect':
        collect_command()
    elif args.command == 'analyze':
        # args.log_file = 'conv2d_nchw.cuda-8-64-32-32-64-3-3-1-1-1-1-cuda.log'
        args.log_file = 'dense_small_batch.cuda-1-512-512-cuda.log'
        analyze_command()
    else:
        raise ValueError("")
