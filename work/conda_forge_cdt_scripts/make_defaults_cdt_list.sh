#!/usr/bin/env bash

DEFAULTS_REPO=${HOME}/Desktop/aggregate

pushd ${DEFAULTS_REPO}
cdt_list=$(compgen  -G "*-cos*" | sort | grep -v "i686")
popd

# rm -rf cdt_list.yaml
# touch cdt_list.yaml
# for cdt in ${cdt_list}; do
#   echo "\"${cdt}\": false" >> cdt_list.yaml
# done
