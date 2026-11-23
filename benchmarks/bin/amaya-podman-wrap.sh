#!/bin/bash
# Runs Amaya's `get-sat` on one input formula inside a podman container.
#
# usage: amaya-podman-wrap.sh <image> <input.smt2> [amaya options...]
#   e.g. amaya-podman-wrap.sh localhost/amaya-bench:cav_2024 f.smt2 --fast -O all
#
# Interface with pycobench (benchmarks/bench/pycobench):
#   * The container is named ${PYCOBENCH_TASK_ID}; the method's
#     `timeout_kill_cmd` removes it with `podman rm -f -t 0`, which sends
#     SIGKILL from the host. Killing this script or the `podman run` client
#     does not stop the container, because the container's processes are
#     children of conmon, not of this script.
#   * CPU time is measured inside the container by bash's `time` keyword with
#     the output format of `time -p`; these are the last three lines of stderr.
#     The method sets `time_measured_by_command: true`, so pycobench does not
#     wrap this script with /usr/bin/time (which would measure only the podman
#     client).
#   * stdout is "result: <stdout of Amaya>"; the exit code is the exit code of
#     `podman run`, i.e., of Amaya. cav_2024 prints "result: <sat|unsat>",
#     devel prints "<sat|unsat>"; a leading "result: " is removed from Amaya's
#     stdout so that both produce the same line.
#   * Every container carries the label `pycobench`; containers left over after
#     an interrupted run are removed by:
#       podman rm -f -t 0 --filter label=pycobench

if [ "$#" -lt 2 ]; then
    echo "usage: ${0} <image> <input.smt2> [amaya options...]" >&2
    exit 1
fi

CONTAINER_IMAGE="$1"
INPUT_FILE_PATH="$(realpath "$2")"
shift 2

CONTAINER_NAME="${PYCOBENCH_TASK_ID:-amaya-bench-$$}"

solver_stdout=$(podman run --rm --pull never \
    --name "${CONTAINER_NAME}" \
    --label pycobench \
    --network none \
    --volume "${INPUT_FILE_PATH}:/input.smt2:ro" \
    --workdir /amaya \
    "${CONTAINER_IMAGE}" \
    bash -c 'TIMEFORMAT=$'"'"'real %3R\nuser %3U\nsys %3S'"'"'; time python3 run-amaya.py "$@" get-sat /input.smt2' \
    bash "$@")
podman_exit_code=$?
solver_stdout="${solver_stdout#result: }"
echo "result: ${solver_stdout}"

exit ${podman_exit_code}
