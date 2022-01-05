import time
import sys
import requests
import cloudpickle
import base64
import backoff

http = "http://127.0.0.1:8000"


@backoff.on_exception(
    backoff.expo,
    requests.exceptions.RequestException,
    max_time=120,
)
def _get_work():
    return requests.get(
        http + "/tasks",
    )


@backoff.on_exception(
    backoff.expo,
    requests.exceptions.RequestException,
    max_time=120,
)
def _send_result(taskid, buff):
    return requests.post(
        http + "/results/" + taskid,
        data=base64.b64encode(buff),
    )


def main(worker_id):
    sleep_time = 1.0

    while True:
        time.sleep(sleep_time)

        res = _get_work()
        if res.status_code == 200:
            data = res.json()
            if data["command"] == "run":

                if "poll_interval" in data:
                    print(
                        "[%s] resetting poll interval: %s -> %s" % (
                            worker_id,
                            sleep_time,
                            data["poll_interval"]
                        ),
                        flush=True,
                    )
                sleep_time = float(data.get("poll_interval", sleep_time))

                if "taskid" in data:
                    print(
                        "[%s] processing task:" % worker_id,
                        data["taskid"],
                        flush=True,
                    )

                    try:
                        rd = cloudpickle.loads(base64.b64decode(data["data"]))
                        res = rd[0](*rd[1], **rd[2])
                    except Exception as e:
                        res = e

                    buff = cloudpickle.dumps(res)
                    res = _send_result(data["taskid"], buff)
                    print(
                        "[%s] send result:" % worker_id,
                        data["taskid"],
                        res.status_code,
                        flush=True,
                    )
            elif data["command"] == "shutdown":
                print(
                    "[%s] shutting down" % worker_id,
                    flush=True,
                )
                sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1])
