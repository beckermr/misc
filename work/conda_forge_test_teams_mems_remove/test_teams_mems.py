import github
import os

from conda_smithy.github import configure_github_team

from ruamel.yaml import YAML

import difflib
import pprint


def diff_objs(d1, d2):
    return ('\n' + '\n'.join(difflib.ndiff(
                   pprint.pformat(d1, width=1).splitlines(),
                   pprint.pformat(d2, width=1).splitlines())))


def assert_feedstock_teams_mems(repo, fs_team, mems, teams):
    curr_mems = set(m.login.lower() for m in fs_team.get_members())
    assert curr_mems == set(mems), diff_objs(curr_mems, set(mems))

    curr_teams = set(t.slug for t in repo.get_teams())
    assert curr_teams == set(teams), diff_objs(curr_teams, set(teams))


class DummyMeta(object):
    def __init__(self, meta_yaml):
        _yml = YAML(typ='jinja2')
        _yml.indent(mapping=2, sequence=4, offset=2)
        _yml.width = 160
        _yml.allow_duplicate_keys = True
        self.meta = _yml.load(meta_yaml)


gh = github.Github(os.environ['GITHUB_TOKEN'])
org = gh.get_organization("conda-forge")
repo = gh.get_repo("conda-forge/cf-autotick-bot-test-package-feedstock")
feedstock_name = "cf-autotick-bot-test-package"
fs_team = org.get_team_by_slug(feedstock_name)
test_team = org.get_team_by_slug("test-team-team")

# set initial team
print("setting initial teams and members...")
meta = DummyMeta(
    """\
extra:
  recipe-maintainers:
    - conda-forge-daemon
    - conda-forge/test-team-team
"""
)
configure_github_team(meta, repo, org, feedstock_name)
assert_feedstock_teams_mems(
    repo, fs_team,
    ["conda-forge-daemon"],
    ["test-team-team", feedstock_name],
)


print("testing adding member...")
meta = DummyMeta(
    """\
extra:
  recipe-maintainers:
    - conda-forge-daemon
    - conda-forge/test-team-team
    - beckermr
"""
)
configure_github_team(meta, repo, org, feedstock_name)
assert_feedstock_teams_mems(
    repo, fs_team,
    ["conda-forge-daemon", "beckermr"],
    ["test-team-team", feedstock_name],
)


print("testing removing member...")
meta = DummyMeta(
    """\
extra:
  recipe-maintainers:
    - conda-forge-daemon
    - conda-forge/test-team-team
"""
)
configure_github_team(meta, repo, org, feedstock_name)
assert_feedstock_teams_mems(
    repo, fs_team,
    ["conda-forge-daemon"],
    ["test-team-team", feedstock_name],
)


print("testing removing team...")
meta = DummyMeta(
    """\
extra:
  recipe-maintainers:
    - conda-forge-daemon
"""
)
configure_github_team(meta, repo, org, feedstock_name)
assert_feedstock_teams_mems(
    repo, fs_team,
    ["conda-forge-daemon"],
    [feedstock_name],
)


print("testing adding team...")
meta = DummyMeta(
    """\
extra:
  recipe-maintainers:
    - conda-forge-daemon
    - conda-forge/test-team-team
"""
)
configure_github_team(meta, repo, org, feedstock_name)
assert_feedstock_teams_mems(
    repo, fs_team,
    ["conda-forge-daemon"],
    ["test-team-team", feedstock_name],
)

print("resetting initial teams and members...")
meta = DummyMeta(
    """\
extra:
  recipe-maintainers:
    - conda-forge-daemon
    - conda-forge/test-team-team
"""
)
configure_github_team(meta, repo, org, feedstock_name)
assert_feedstock_teams_mems(
    repo, fs_team,
    ["conda-forge-daemon"],
    ["test-team-team", feedstock_name],
)
