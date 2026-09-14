#!/usr/bin/env bash
# Run a profiling command inside the DTK container on a DCU host,
# with GPU devices, the host hyhal runtime, and rocprof on PATH.
#
# Usage:
#   ./profile_container.sh <workdir-on-host> <command...>
# Example:
#   ./profile_container.sh /tmp/run1 "rocprof -i prof_input.txt -o results.csv ./harness"
#
# The host workdir is mounted at /work (writable — hipify and rocprof outputs
# need writable paths; see reference/09-common-issues.md #5).

set -euo pipefail

IMAGE="${DTK_IMAGE:-harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.29.0-ubuntu22.04-dtk26.04-py3.10-20260831-qwen3.8flashnext}"
WORKDIR_HOST="${1:?usage: profile_container.sh <host-workdir> <command...>}"
shift

docker run --rm \
  --device /dev/dri --device /dev/kfd \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "$WORKDIR_HOST":/work \
  -w /work \
  "$IMAGE" \
  bash -c "source /opt/dtk/env.sh; export PATH=/opt/dtk/rocprofiler/bin:\$PATH; $*"
