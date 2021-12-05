#!/bin/bash

for i in `seq 1 4`; do
    sleep 30
    python pctest.py 2000 &
done
