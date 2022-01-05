from fastapi import Request, FastAPI

app = FastAPI()

TASKS = {}
RESULTS = {}
STATE = "run"


@app.get("/")
async def root():
    return {"tasks": len(TASKS), "results": len(RESULTS)}


@app.post("/commands/{cmd}")
async def command(cmd: str):
    global STATE
    STATE = cmd


@app.post("/tasks/{taskid}")
async def post_tasks(taskid: str, request: Request):
    data = await request.body()
    print("got taskid:", taskid, flush=True)
    TASKS[taskid] = data


@app.get("/tasks")
async def get_tasks():
    if TASKS:
        tid = list(TASKS.keys())[0]
        data = TASKS.pop(tid)
        print("sent taskid:", tid, flush=True)
        return {"taskid": tid, "data": data, "command": "run"}
    else:
        return {"command": STATE}


@app.post("/results/{taskid}")
async def post_results(taskid: str, request: Request):
    data = await request.body()
    print("got result taskid:", taskid, flush=True)
    RESULTS[taskid] = data


@app.get("/results")
async def get_results():
    if RESULTS:
        tid = list(RESULTS.keys())[0]
        data = RESULTS.pop(tid)
        print("sent result askid:", tid, flush=True)
        return {"taskid": tid, "data": data}
    else:
        return {}
