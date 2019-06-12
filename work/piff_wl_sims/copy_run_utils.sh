#!/usr/bin/env bash

for dr in `ls -1d v*`; do
    echo $dr
    cp run_utils/* $dr/.
done
