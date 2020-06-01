#!/bin/bash

set -e

rm -rf cf-autotick-bot-test-package-feedstock*

export CONDA_SMITHY_LOGLEVEL=DEBUG

git clone https://github.com/conda-forge/cf-autotick-bot-test-package-feedstock.git
pushd cf-autotick-bot-test-package-feedstock

git checkout master
python ../prep_pr.py
echo "rerendering..."
conda smithy rerender
git commit --allow-empty -am '[ci skip] [cf admin skip] azure only rerend'
git push

for bn in `seq $1 $2`; do
  python ../make_prs.py $bn
done

popd
rm -rf cf-autotick-bot-test-package-feedstock
