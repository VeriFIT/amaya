#!/bin/bash
# Runs cvc5 on one input formula inside a podman container, optionally after converting the formula with
# `amaya convert` in the same container.
#
# usage: cvc5-podman-wrap.sh <image> <input.smt2>
#            cvc5 on the input formula.
#        cvc5-podman-wrap.sh <image> <input.smt2> --amaya-convert [amaya options...]
#            `python3 run-amaya.py [amaya options...] convert /input.smt2`, then cvc5 on its SMT2 output.
#   e.g. cvc5-podman-wrap.sh localhost/amaya-bench:cvc5 f.smt2 --amaya-convert -O all --opt-fixpoint
#
# Interface with pycobench (benchmarks/bench/pycobench), the same as benchmarks/bin/amaya-podman-wrap.sh:
#   * The container is named ${PYCOBENCH_TASK_ID}; the method's `timeout_kill_cmd` removes it with
#     `podman rm -f -t 0`.
#   * CPU time is measured inside the container by bash's `time` keyword with the output format of `time -p`
#     (the last three lines of stderr). With --amaya-convert, the measured time is the sum of the conversion
#     and of cvc5. The method sets `time_measured_by_command: true`.
#   * stdout is "result: <first line of cvc5's stdout>". If the conversion fails, stdout is
#     "result: convert-error" and the exit code is that of `amaya convert`; cvc5 is not run.
#     Otherwise the exit code is that of cvc5.
#   * Every container carries the label `pycobench`; leftover containers are removed by:
#       podman rm -f -t 0 --filter label=pycobench

if [ "$#" -lt 2 ]; then
    echo "usage: ${0} <image> <input.smt2> [--amaya-convert amaya options...]" >&2
    exit 1
fi

CONTAINER_IMAGE="$1"
INPUT_FILE_PATH="$(realpath "$2")"
shift 2

CONTAINER_NAME="${PYCOBENCH_TASK_ID:-cvc5-bench-$$}"

if [ "${1:-}" = "--amaya-convert" ]; then
    shift
    container_script='
        run_conversion_then_cvc5() {
            python3 run-amaya.py "$@" convert /input.smt2 > /tmp/converted.smt2 || {
                conversion_exit_code=$?; echo convert-error; return ${conversion_exit_code}; }
            cvc5 --lang smt2 /tmp/converted.smt2
        }
        TIMEFORMAT=$'"'"'real %3R\nuser %3U\nsys %3S'"'"'
        time run_conversion_then_cvc5 "$@"'
else
    container_script='
        TIMEFORMAT=$'"'"'real %3R\nuser %3U\nsys %3S'"'"'
        time cvc5 --lang smt2 /input.smt2'
fi

solver_stdout=$(podman run --rm --pull never \
    --name "${CONTAINER_NAME}" \
    --label pycobench \
    --network none \
    --volume "${INPUT_FILE_PATH}:/input.smt2:ro" \
    --workdir /amaya \
    "${CONTAINER_IMAGE}" \
    bash -c "${container_script}" \
    bash "$@")
podman_exit_code=$?
echo "result: $(head -1 <<< "${solver_stdout}")"

exit ${podman_exit_code}
