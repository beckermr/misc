import sys
import time
import uuid
import requests
import joblib
import io
import base64
import cloudpickle

http = "http://127.0.0.1:8000"

while True:
    time.sleep(float(sys.argv[1]))
    tid = uuid.uuid4().hex
    print("sending task:", tid, flush=True)
    buff = io.BytesIO()

    def _do_this(a):
        return a.upper()

    buff = cloudpickle.dumps(joblib.delayed(_do_this)("blah"))

    requests.post(
        http + "/tasks/" + uuid.uuid4().hex,
        data=base64.b64encode(buff),
    )

    res = requests.get(
        http + "/results",
    )
    if res.status_code == 200:
        buff = res.raw.data
        if len(buff) > 0:
            rd = cloudpickle.loads(base64.b64decode(buff))
            print("got result:", rd, flush=True)
