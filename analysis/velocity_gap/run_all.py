"""Dispatch all velocity-gap + mAD-sweep jobs over GPUs (one process per GPU, serial queue).

Conditions:
  noise axis : ckpt_epoch_300, t0 in 0.1..0.9
  budget axis: ckpt_epoch_{5,10,20,40,80,150,300}, t0 = 0.1
Each condition runs measure_vgap (K=20 trajectory) and sweep_mad (K sweep).
"""
import itertools, os, queue, subprocess, sys, threading

SP = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get('REPO_ROOT', os.path.abspath(os.path.join(SP, '..', '..')))
RUN = os.environ.get('RUN_DIR', os.path.join(REPO, 'logs/mvtec_dit_fm'))
CACHE = os.environ.get('VGAP_CACHE', os.path.join(SP, 'cache'))
OUT_VGAP = os.path.join(RUN, 'velocity_gap')
OUT_MAD = os.path.join(RUN, 'mad_sweeps')
PY = sys.executable
KS = [1, 2, 4, 8, 20, 40, 80, 200]

NOISE_T0 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
BUDGET_EPOCHS = [5, 10, 20, 40, 80, 150, 300]
GPUS = [0, 1, 2, 3, 4, 5, 6, 7]


def make_jobs(seeds=(0,), mad_only=False):
    jobs = []
    conds = [(300, t0) for t0 in NOISE_T0]
    conds += [(ep, 0.1) for ep in BUDGET_EPOCHS if ep != 300]
    for seed in seeds:
        for ep, t0 in conds:
            ckpt = os.path.join(RUN, f'checkpoints/ckpt_epoch_{ep}.pth')
            tag = f'ep{ep}'
            if not mad_only:
                jobs.append([PY, os.path.join(SP, 'measure_vgap.py'), '--cache', CACHE,
                             '--ckpt', ckpt, '--t0', str(t0), '--steps', '20',
                             '--out', OUT_VGAP, '--tag', tag, '--seed', str(seed)])
            jobs.append([PY, os.path.join(SP, 'sweep_mad.py'), '--cache', CACHE,
                         '--ckpt', ckpt, '--t0', str(t0), '--ks', *map(str, KS),
                         '--out', OUT_MAD, '--tag', tag, '--seed', str(seed)])
    return jobs


def worker(gpu, q, logdir):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    while True:
        try:
            i, cmd = q.get_nowait()
        except queue.Empty:
            return
        name = f'job{i:02d}_{os.path.basename(cmd[1]).split(".")[0]}_{cmd[7 if "measure" in cmd[1] else 7]}'
        log = os.path.join(logdir, f'job{i:02d}_gpu{gpu}.log')
        with open(log, 'w') as f:
            f.write(' '.join(cmd) + '\n\n')
            f.flush()
            r = subprocess.run(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT)
        print(f'[gpu{gpu}] job{i:02d} exit={r.returncode} ({" ".join(cmd[3:])})', flush=True)


def main():
    os.makedirs(OUT_VGAP, exist_ok=True)
    os.makedirs(OUT_MAD, exist_ok=True)
    logdir = os.path.join(RUN, 'velocity_gap', 'joblogs')
    os.makedirs(logdir, exist_ok=True)
    if len(sys.argv) > 1 and sys.argv[1] == 'seeds':
        jobs = make_jobs(seeds=(1, 2), mad_only=True)
    else:
        jobs = make_jobs()
    q = queue.Queue()
    for i, j in enumerate(jobs):
        q.put((i, j))
    threads = [threading.Thread(target=worker, args=(g, q, logdir)) for g in GPUS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print('ALL JOBS DONE')


if __name__ == '__main__':
    main()
