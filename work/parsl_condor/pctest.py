import parsl
from parsl.providers import CondorProvider
from parsl.config import Config
from parsl.executors import HighThroughputExecutor

from parsl.configs.local_threads import config

SCHED_OPTS = ""
WORKER_INIT = ""
WALLTIME = ""

condor_config = Config(
    executors=[
        HighThroughputExecutor(
            worker_debug=True,
            provider=CondorProvider(
                cores_per_slot=1,
                mem_per_slot=2,
                nodes_per_block=1,
                init_blocks=10,
                max_blocks=10,
                scheduler_options=SCHED_OPTS,
                worker_init=WORKER_INIT,
                walltime=WALLTIME
            )
        )
    ],
)

parsl.load(condor_config)


@parsl.python_app
def app_double(x):
    return x*2


@parsl.python_app
def app_sum(inputs=[]):
    return sum(inputs)


items = range(0, 1000)

mapped_results = []
for i in items:
    x = app_double(i)
    mapped_results.append(x)

total = app_sum(inputs=mapped_results)

print(total.result())
