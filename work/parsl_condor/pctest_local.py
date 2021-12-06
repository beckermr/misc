import time
import sys
import parsl
from parsl.providers import LocalProvider
from parsl.config import Config
from parsl.executors import HighThroughputExecutor

import uuid

ID = uuid.uuid4().hex

START = time.time()

SCHED_OPTS = """
kill_sig        = SIGINT
+Experiment     = "astro"
Notification    = Never
Image_Size       =  1000000
"""
# GetEnv          = True

WORKER_INIT = """

source ~/.bashrc

export OMP_NUM_THREADS=1

if [[ -n $_CONDOR_SCRATCH_DIR ]]; then
    # the condor system creates a scratch directory for us,
    # and cleans up afterward
    tmpdir=$_CONDOR_SCRATCH_DIR
    export TMPDIR=$tmpdir
else
    # otherwise use the TMPDIR
    tmpdir='.'
    mkdir -p $tmpdir
fi

source activate bnl

echo `which python`
"""

WALLTIME = "48:00:00"

condor_config = Config(
    strategy='htex_auto_scale',
    executors=[
        HighThroughputExecutor(
            worker_debug=True,
            max_workers=1,
            poll_period=5000,
            provider=LocalProvider(
                parallelism=1,
                min_blocks=1,
                max_blocks=16,
                worker_init=WORKER_INIT,
            )
        )
    ],
)

parsl.load(condor_config)


@parsl.python_app
def app_double(x):
    import time
    time.sleep(30)
    return 2*x


items = range(0, int(sys.argv[1]))

mapped_results = []
for i in items:
    x = app_double(i)
    mapped_results.append(x)


res = []
n_tot = len(mapped_results)
n_left = len(mapped_results)
start = time.time()
eta = -1
while n_left > 0:
    for i in range(len(mapped_results)):
        f = mapped_results[i]
        if n_left == 1:
            print("[%s] done?" % ID, f.done(), flush=True)
        try:
            f.result(timeout=10)
        except Exception:
            pass
        if n_left == 1:
            print("[%s] done?" % ID, f.done(), flush=True)
        if f.done():
            res.append(f.result())
            n_left -= 1
            eta = (time.time() - start) / (n_tot - n_left) * n_left
            print(
                "[%s - %06ds] eta|done|left:" % (
                    ID, int(time.time() - START)
                ), eta, n_tot - n_left, n_left, flush=True,
            )
            mapped_results.pop(i)
            break

# for f in as_completed(mapped_results):
#    n_left -= 1

print(
    "[%s] total:" % ID,
    sum(res), sum(2*x for x in range(0, int(sys.argv[1]))), flush=True,
)

print("[%s] done w/ loop:" % ID, time.time() - start, flush=True)

# total = app_sum(inputs=mapped_results)
# print("total:", total.result(), flush=True)
