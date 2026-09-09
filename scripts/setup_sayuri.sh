#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sayuri_source="${SAYURI_SOURCE:-${repository_root}/UsedCode/Sayuri}"
sayuri_build="${SAYURI_BUILD_DIR:-${repository_root}/.external/sayuri-build}"
weight_dir="${SAYURI_WEIGHT_DIR:-${repository_root}/.external/sayuri-weights}"
weight_name="sayuri-b12xc384nbt-s3592000-c4310000-w426561-swa.bin.txt"
weight_id="18ll3DgawmUPPDhiZNDLRQjTGCJxD1cU4"
weight_sha256="31e31e6b8c3af59bc996470c91c262ea9aeb4fb1df6c8d2d5fafc320240bcb81"
download_weights=0

for argument in "$@"; do
    case "${argument}" in
        --download-weights) download_weights=1 ;;
        *) echo "Unknown argument: ${argument}" >&2; exit 2 ;;
    esac
done

if [[ ! -f "${sayuri_source}/CMakeLists.txt" ]]; then
    echo "Sayuri source is missing at ${sayuri_source}" >&2
    echo "Clone https://github.com/CGLemon/Sayuri into UsedCode/Sayuri first." >&2
    exit 1
fi

git -C "${sayuri_source}" submodule update --init --recursive --depth 1
cmake -S "${sayuri_source}" -B "${sayuri_build}" \
    -DBLAS_BACKEND=EIGEN \
    -DUSE_FAST_PARSER=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${sayuri_build}" -j "${SAYURI_BUILD_JOBS:-4}"

if [[ "${download_weights}" -eq 1 ]]; then
    mkdir -p "${weight_dir}"
    weight_path="${weight_dir}/${weight_name}"
    if [[ ! -f "${weight_path}" ]]; then
        curl --fail --location --retry 3 \
            --output "${weight_path}" \
            "https://drive.usercontent.google.com/download?id=${weight_id}&export=download&confirm=t"
    fi
    echo "${weight_sha256}  ${weight_path}" | sha256sum --check --status || {
        echo "Downloaded Sayuri weight checksum mismatch" >&2
        exit 1
    }
    echo "Sayuri weights: ${weight_path}"
fi

echo "Sayuri executable: ${sayuri_build}/sayuri"
