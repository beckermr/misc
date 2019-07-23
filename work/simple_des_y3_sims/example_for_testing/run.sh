#!/usr/bin/env bash

run-simple-des-y3-sim \
  prep \
  --tilename="DES2122+0001" \
  --bands="riz" \
  --output-desdata="outputs"

run-simple-des-y3-sim \
  galsim \
  --tilename="DES2122+0001" \
  --bands="r" \
  --output-desdata="outputs" \
  --config-file="config.yaml" \
  --seed=12

run-simple-des-y3-sim \
  true-detection \
  --tilename="DES2122+0001" \
  --bands="r" \
  --output-desdata="outputs" \
  --config-file="config.yaml"

run-simple-des-y3-sim \
  meds \
  --tilename="DES2122+0001" \
  --bands="r" \
  --output-desdata="outputs" \
  --config-file="config.yaml" \
  --meds-config-file="../default_configs/meds.yaml"

run-simple-des-y3-sim \
  metacal \
  --tilename="DES2122+0001" \
  --bands="r" \
  --output-desdata="outputs" \
  --seed=67
