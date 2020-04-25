#!/bin/bash

set -e

export CONDA_SMITHY_LOGLEVEL=DEBUG

# pushd cf-test-master
# for linux in circle travis azure; do
#   for osx in circle travis azure; do
#     git co master
#     git co -b ${linux}-${osx}
#     git push --set-upstream origin ${linux}-${osx}
#   done
# done
# popd

pushd cf-autotick-bot-test-package-feedstock
for linux in circle travis azure; do
  for osx in circle travis azure; do
    python ../make_prs.py ${linux} ${osx} $1
  done
done
