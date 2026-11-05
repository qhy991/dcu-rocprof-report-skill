#!/usr/bin/env python3
"""Verify that a failed compiler-environment probe reaps its blocked child."""
import json
import os
import pathlib
import subprocess
import sys

out = pathlib.Path(sys.argv[1])
cache = pathlib.Path(sys.argv[2])
out.mkdir(exist_ok=False)
check = pathlib.Path(__file__).with_name("check_compiler_env.py")
code = '''
import pathlib, re, runpy, sys
old = pathlib.Path.read_text
def fail_target_maps(self, *args, **kwargs):
    if re.fullmatch(r"/proc/[0-9]+/maps", str(self)):
        raise RuntimeError("forced compiler maps inspection failure")
    return old(self, *args, **kwargs)
pathlib.Path.read_text = fail_target_maps
sys.argv = [sys.argv[1], sys.argv[2], sys.argv[3]]
runpy.run_path(sys.argv[0], run_name="__main__")
'''
child = subprocess.Popen(
    [sys.executable, "-c", code, str(check), str(out / "test-artifacts"), str(cache)],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
)
try:
    stdout, stderr = child.communicate(timeout=60)
except subprocess.TimeoutExpired:
    child.kill()
    child.communicate()
    raise
if child.returncode == 0 or b"forced compiler maps inspection failure" not in stderr:
    raise SystemExit("The injected failure did not occur as expected.")
if (out / "test-artifacts" / "acceptance.json").exists():
    raise SystemExit("A failed run must not produce an acceptance file.")
log = pathlib.Path(os.environ["OUTPUT_PATH"]).parent / "compiler-environments" / f"{child.pid}.jsonl"
events = [json.loads(line) for line in log.read_text().splitlines()]
spawned = [e["child_pid"] for e in events if e.get("event") == "compiler_spawned_with_profiler_removed"]
if len(spawned) < 2:
    raise SystemExit("Did not observe both the setup and blocked compiler processes.")
if pathlib.Path(f"/proc/{spawned[-1]}").exists():
    raise SystemExit("The compiler child was not reaped.")
result = {
    "accepted": True,
    "scope": "CPU-only injected failure; no GPU work.",
    "expected_failure_observed": True,
    "failed_run_has_no_acceptance_file": True,
    "blocked_compiler_reaped": True,
}
(out / "failure-acceptance.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
