#!/bin/bash

set -e

rm -rf cf-autotick-bot-test-package-feedstock*

export CONDA_SMITHY_LOGLEVEL=DEBUG

git clone https://github.com/conda-forge/cf-autotick-bot-test-package-feedstock.git
pushd cf-autotick-bot-test-package-feedstock

for branch in `git branch -a`; do
  if [[ ${branch} == *hash_* ]]; then
    echo `basename $branch`
    git push origin --delete `basename $branch`
  fi
done

popd
rm -rf cf-autotick-bot-test-package-feedstock
