#!/bin/bash
# Builds the podman images used by benchmarks/bench/amaya-podman.yaml.
#
# usage: build-images.sh [cav_2024] [devel]   (default: both)
#
# Each image is built from `git archive <revision>`, i.e., from committed
# content only; untracked and modified files in the working tree are not part
# of the image.
#   amaya-bench:cav_2024  <- tag cav_2024, benchmarks/containers/Containerfile.cav_2024
#   amaya-bench:devel     <- branch devel, the Dockerfile at the root of that revision
# The source commit is stored in the image label `amaya.revision`.
set -euo pipefail

REPOSITORY_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
CONTAINERS_DIRECTORY="${REPOSITORY_ROOT}/benchmarks/containers"

build_image_from_revision() {
    local revision="$1" image_tag="$2" containerfile_path="$3"
    local build_context_directory
    build_context_directory="$(mktemp -d)"
    trap 'rm -rf "${build_context_directory}"' RETURN
    local commit_hash
    commit_hash="$(git -C "${REPOSITORY_ROOT}" rev-parse "${revision}^{commit}")"
    git -C "${REPOSITORY_ROOT}" archive "${commit_hash}" | tar -x -C "${build_context_directory}"
    if [ -z "${containerfile_path}" ]; then
        containerfile_path="${build_context_directory}/Dockerfile"
    fi
    echo ">>> building ${image_tag} from ${revision} (${commit_hash})"
    podman build \
        --label "amaya.revision=${commit_hash}" \
        -t "${image_tag}" \
        -f "${containerfile_path}" \
        "${build_context_directory}"
}

requested_images=("$@")
if [ "${#requested_images[@]}" -eq 0 ]; then
    requested_images=(cav_2024 devel)
fi

for requested_image in "${requested_images[@]}"; do
    case "${requested_image}" in
        cav_2024) build_image_from_revision cav_2024 amaya-bench:cav_2024 "${CONTAINERS_DIRECTORY}/Containerfile.cav_2024" ;;
        devel)    build_image_from_revision devel    amaya-bench:devel    "" ;;
        *) echo "unknown image: ${requested_image}" >&2; exit 1 ;;
    esac
done
