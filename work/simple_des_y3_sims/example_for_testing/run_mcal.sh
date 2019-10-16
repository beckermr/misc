#!/usr/bin/env bash

run-simple-des-y3-sim-mcal \
  --tilename="DES2122+0001" \
  --bands="r" \
  --output-desdata="outputs" \
  --mcal-config-file="../default_configs/run-mcal-nouberseg.yaml" \
  --seed=34
