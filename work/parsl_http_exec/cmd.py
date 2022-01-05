import sys
import requests

http = "http://127.0.0.1:8000"

requests.post(
    http + "/commands/" + sys.argv[1],
)
