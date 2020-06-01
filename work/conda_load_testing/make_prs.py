#!/user/bin/env python
import sys
import subprocess
import uuid

META = """\
{% set name = "cf-autotick-bot-test-package" %}
{% set version = "0.9" %}

package:
  name: {{ name|lower }}
  version: {{ version }}

source:
  url: https://github.com/regro/cf-autotick-bot-test-package/archive/v{{ version }}.tar.gz
  sha256: 74d5197d4ca8afb34b72a36fc8763cfaeb06bdbc3f6d63e55099fe5e64326048

build:
  number: {{ build }}
  string: "{{ cislug }}_py{{ py }}h{{ PKG_HASH }}_{{ build }}"

requirements:
  host:
    - python
    - pip
  run:
    - python

test:
  commands:
    - echo "works!"

about:
  home: https://github.com/regro/cf-scripts
  license: BSD-3-Clause
  license_family: BSD
  license_file: LICENSE
  summary: testing feedstock for the regro-cf-autotick-bot

extra:
  recipe-maintainers:
    - beckermr
    - conda-forge/bot
"""  # noqa

build = int(sys.argv[1])
BUILD_SLUG = "{% set build = " + str(build) + " %}\n"

h = uuid.uuid4().hex[0:6]

CI_SLUG = '{% set cislug = "' + h + '" %}\n'

BRANCH = "hash_" + h + "_" + str(build)

print("\n\n=========================================")
print("making the head branch")
subprocess.run(
    ["git", "checkout", "master"],
    check=True,
)

subprocess.run(
    ["git", "checkout", "-b", BRANCH],
    check=True,
)

print("\n\n=========================================")
print("editing the recipe")

with open("recipe/meta.yaml", "w") as fp:
    fp.write(CI_SLUG)
    fp.write(BUILD_SLUG)
    fp.write(META)


subprocess.run(
    ["git", "add", "recipe/meta.yaml"],
    check=True,
)

print("\n\n=========================================")
print("commiting")

subprocess.run(
    ["git", "commit", "-am", "'[cf admin skip] test %s %s'" % (h, build)],
    check=True
)

print("\n\n=========================================")
print("pushing to upstream")

subprocess.run(
    ["git", "push", "--set-upstream", "origin", BRANCH]
)
