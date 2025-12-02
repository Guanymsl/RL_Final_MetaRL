import os
import torch as tc

def _format_name(kind, steps):
    filename = f"{kind}_{steps}.pth"
    return filename

def _parse_name(filename):
    kind, steps = filename.split(".")[0].split("_")
    steps = int(steps)
    return {
        "kind": kind,
        "steps": steps
    }

def _latest_n_checkpoint_steps(base_path, n=5):
    steps = set(map(lambda x: _parse_name(x)['steps'], os.listdir(base_path)))
    latest_steps = sorted(steps)
    latest_n = latest_steps[-n:]
    return latest_n

def _latest_step(base_path):
    return _latest_n_checkpoint_steps(base_path, n=1)[-1]

def save_checkpoint(
        steps,
        checkpoint_dir,
        model_name,
        model,
        optimizer,
        scheduler
    ):
    base_path = os.path.join(checkpoint_dir, model_name)
    os.makedirs(base_path, exist_ok=True)

    model_path = os.path.join(base_path, _format_name('model', steps))
    optim_path = os.path.join(base_path, _format_name('optimizer', steps))
    sched_path = os.path.join(base_path, _format_name('scheduler', steps))

    tc.save(model.state_dict(), model_path)
    tc.save(optimizer.state_dict(), optim_path)
    if scheduler is not None:
        tc.save(scheduler.state_dict(), sched_path)

    latest_n_steps = _latest_n_checkpoint_steps(base_path, n=5)
    for file in os.listdir(base_path):
        if _parse_name(file)['steps'] not in latest_n_steps:
            os.remove(os.path.join(base_path, file))

def maybe_load_checkpoint(
        checkpoint_dir,
        model_name,
        model,
        optimizer,
        scheduler,
        steps
    ):
    base_path = os.path.join(checkpoint_dir, model_name)
    try:
        if steps is None:
            steps = _latest_step(base_path)

        model_path = os.path.join(base_path, _format_name('model', steps))
        optim_path = os.path.join(base_path, _format_name('optimizer', steps))
        sched_path = os.path.join(base_path, _format_name('scheduler', steps))

        model.load_state_dict(tc.load(model_path))
        optimizer.load_state_dict(tc.load(optim_path))
        if scheduler is not None:
            scheduler.load_state_dict(tc.load(sched_path))

        print(f"Loaded checkpoint from {base_path}, with step {steps}.")
        print("Continuing from checkpoint.")
    except FileNotFoundError:
        print(f"Bad checkpoint or none at {base_path} with step {steps}.")
        print("Running from scratch.")
        steps = 0

    return steps
