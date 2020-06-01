#!/bin/bash

set -e

rm -rf cf-autotick-bot-test-package-feedstock*
rm -rf cf-test-master*

export CONDA_SMITHY_LOGLEVEL=DEBUG

git clone https://github.com/conda-forge/cf-autotick-bot-test-package-feedstock.git cf-test-master
pushd cf-test-master
for linux in azure; do
  for osx in azure; do
    git co master
    git co -b ${linux}-${osx}-$1
    git commit --allow-empty -am '[ci skip] branch for testing'
    git push --set-upstream origin ${linux}-${osx}-$1
  done
done
popd
rm -rf cf-test-master

git clone https://github.com/beckermr/cf-autotick-bot-test-package-feedstock.git cf-autotick-bot-test-package-feedstock-$1
pushd cf-autotick-bot-test-package-feedstock-$1

git remote add upstream https://github.com/conda-forge/cf-autotick-bot-test-package-feedstock.git

# git branch -r | grep -v '\->' | while read remote; do git branch --track "${remote#origin/}" "$remote"; done
git fetch --all
git pull --all

for linux in azure; do
  for osx in azure; do
    python ../make_prs.py ${linux} ${osx} $1
  done
done

popd
rm -rf cf-autotick-bot-test-package-feedstock-$1
