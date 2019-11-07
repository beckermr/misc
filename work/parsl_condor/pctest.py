import os
from parsl.providers import CondorProvider
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.addresses import address_by_hostname
proxy = '/tmp/x509up_u{}'.format(os.getuid())
if not os.path.isfile(proxy):
   raise RuntimeError('No valid proxy found-- please run voms-proxy-init -voms cms')
worker_init = '''
source /cvmfs/sft.cern.ch/lcg/views/LCG_95apython3/x86_64-centos7-gcc7-opt/setup.sh
export PATH=~/.local/bin:$PATH
export PYTHONPATH=~/.local/lib/python3.6/site-packages:$PYTHONPATH
export X509_USER_PROXY=pwd/{}
'''.format(os.path.basename(proxy))
config = Config(
   executors=[
       HighThroughputExecutor(
           address=address_by_hostname(),
           cores_per_worker=1,
           worker_debug=True,
           provider=CondorProvider(
               cores_per_slot=8,
               init_blocks=80,
               max_blocks=80,
               worker_init=worker_init,
               transfer_input_files=[proxy],
           ),
       )
   ],
   retries=5,
   strategy=None
)
