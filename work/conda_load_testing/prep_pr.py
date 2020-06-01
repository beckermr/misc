#!/user/bin/env python
import subprocess
import yaml

with open("conda-forge.yml", "r") as fp:
    cfg = yaml.safe_load(fp)

if "aarch64" in cfg["provider"]:
    del cfg["provider"]["aarch64"]

cfg["provider"]["linux"] = "azure"
cfg["provider"]["osx"] = "azure"
cfg["provider"]["win"] = "azure"
cfg["provider"]["linux_ppc64le"] = "azure"
cfg["provider"]["linux_aarch64"] = "azure"
cfg["conda_forge_output_validation"] = True

with open("conda-forge.yml", "w") as fp:
    yaml.dump(cfg, fp)

with open("recipe/conda_build_config.yaml", "w") as fp:
    fp.write("""\
python:
 - 3.6.* *_cpython
""")

subprocess.run(
    ["git", "add", "conda-forge.yml",
     "recipe/conda_build_config.yaml"],
    check=True,
)
