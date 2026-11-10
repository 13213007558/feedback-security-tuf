# Feedback task starting environment

Upstream: https://github.com/theupdateframework/python-tuf
Upstream commit: `31bba85aba12115eaf3715768195adff1c0d0e49`
Prepared on macOS with CPython 3.12.11. No external database, queue, or cloud account is required.
The dependency file pins the installed runtime and test packages for the selected baseline. It is not a claim that all optional upstream test suites are installed.

From the repository root:

```sh
export NO_INTERNET=1 TZ=UTC
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r .feedback/requirements.txt -e .
.venv/bin/python -m pytest -q -o addopts= tests/test_trusted_metadata_set.py tests/test_updater_delegation_graphs.py tests/test_updater_key_rotations.py tests/test_updater_top_level_update.py tests/test_api.py
```

Scope note: The scope is the command below, not all optional upstream integrations.

Observed baseline (2026-11-10): `111 passed, 2 warnings, 52 subtests passed in 2.91s`
Whole-command wall time: 3.254 seconds, exit code 0. This is one local measurement, not a runtime guarantee. The first install is excluded.

This seed adds preparation metadata only. Application sources and upstream tests are unchanged; there is no task solution in this commit.
