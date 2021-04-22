import numpy as np
from tvm import autotvm
from tvm import topi, te, relay
import tvm
import argparse

import custom

parser = argparse.ArgumentParser()
parser.add_argument('-task_type', type=str, choices=['conv2d', 'dense', 'bmm'], default='dense')
parser.add_argument('-args', type=str, default='1-8-8')
parser.add_argument('-target', type=str, choices=['llvm', 'cuda'], default='llvm')
parser.add_argument('-n_trial', type=int, default=1000)
parser.add_argument('-log_file', type=str, default='')

args = parser.parse_args()


def collect_running_data(task: autotvm.task.Task, log_file, n_trial=-1):
    n_total = len(task.config_space)
    if n_trial == -1:
        n_trial = n_total
    if n_trial >= n_total:
        n_trial = n_total
        tuner_name = 'grid_search'
    else:
        tuner_name = 'random'

    summary = [
        f'Task name: {task.name}',
        f'Args: {task.args}',
        f'Target: {task.target}',
        f'Tuner name: {tuner_name}',
        f'Num of trials: {n_trial} / {n_total} ({n_trial / n_total * 100:.2f}%)',
        f'Log file: {log_file}',
        f'Schedule space:',
        str(task.config_space)
    ]
    with open(f'{log_file}.meta', 'w') as f:
        f.write("\n".join(summary))
    print("\n".join(summary))

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
        ],
    )


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


def conv2d_task(batch_size, in_channels, h, w, out_channels, kernel, padding, strides, groups, target):
    x = relay.var('input', shape=(batch_size, in_channels, h, w))
    w = relay.var('w', shape=(out_channels, in_channels//groups, kernel[0], kernel[1]))
    x = relay.nn.conv2d(x, w, padding=padding, strides=strides, groups=groups)
    task = task_from_relay(x, target, "nn.conv2d")
    return task


def dense_task(batch_size, in_channels, out_channels, target):
    x = relay.var('input', shape=(batch_size, in_channels))
    w = relay.var('w', shape=(out_channels, in_channels))
    x = relay.nn.dense(x, w)
    task = task_from_relay(x, target, "nn.dense")
    return task


def batch_matmul(batch_size, n, m, k, target):
    x = relay.var('x', shape=(batch_size, n, k))
    y = relay.var('y', shape=(batch_size, m, k))
    out = relay.nn.batch_matmul(x, y)
    task = task_from_relay(out, target, "nn.batch_matmul")
    return task


def main():
    if args.task_type == 'conv2d':
        bs, ic, h, w, oc, kx, ky, px, py, sx, sy, g = [int(v) for v in args.args.split('-')]
        task = conv2d_task(batch_size=bs, in_channels=ic, h=h, w=w,
                           out_channels=oc, kernel=(kx, ky), padding=(px, py), strides=(sx, sy), groups=g,
                           target=args.target)
    elif args.task_type == 'dense':
        bs, ic, oc = [int(v) for v in args.args.split('-')]
        task = dense_task(batch_size=bs, in_channels=ic, out_channels=oc, target=args.target)
    elif args.task_type == 'bmm':
        bs, n, m, k = [int(v) for v in args.args.split('-')]
        task = batch_matmul(batch_size=bs, n=n, m=m, k=k, target=args.target)
    else:
        raise ValueError("")
    log_file = args.log_file if len(args.log_file) > 0 else f'{args.task_type}-{args.args}-{args.target}.log'
    collect_running_data(task, log_file=log_file, n_trial=args.n_trial)


if __name__ == '__main__':
    main()
