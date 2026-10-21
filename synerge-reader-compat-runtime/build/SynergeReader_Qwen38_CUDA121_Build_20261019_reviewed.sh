#!/usr/bin/env bash
# Experimental, separate CUDA backend build. Does not modify the production service.
# Builds only the CUDA module from the exact llama.cpp revision in Ollama 0.32.12.
# Discovers GPU 0 without loading any model. A positive result is discovery only.
set -euo pipefail
umask 077

BASE=/home/haihua/synergereader-model-tests
RUNTIME='ollama/ollama@sha256:07e04ed05eb2232d66060e082da9ee6db9f37e9bf6d4260acb2535a6fd8e6ea7'
BUILDER='nvidia/cuda@sha256:327c9e046fbf662275be0934742f7e5412f9b24402ee90bf4d649c1a21707912'
OLLAMA_COMMIT=e0e1d3c7d960f5879386871deb23582800c8599d
LLAMA_COMMIT=0b1bad14ff204627636aeb1de22ddcd5acb859d4

for sr_cmd in docker curl python3 timeout nvidia-smi sha256sum; do
    command -v "$sr_cmd" >/dev/null
done
test "$(uname -m)" = x86_64
docker image inspect "$RUNTIME" >/dev/null
GPU_UUID="$(nvidia-smi --id=0 --query-gpu=uuid --format=csv,noheader)"
[[ "$GPU_UUID" =~ ^GPU-[0-9a-fA-F-]+$ ]]
# Ollama logs the PCI bus id (pci_id=), never the UUID; use it for the pass check.
GPU_PCI="$(nvidia-smi --id="$GPU_UUID" --query-gpu=pci.bus_id --format=csv,noheader)"
[[ "$GPU_PCI" =~ ^[0-9A-Fa-f]{4,8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]$ ]]
# The builder image (~7.5 GiB unpacked), build cache and new image land in the
# Docker root shared with production containers. Refuse to start without headroom.
DOCKER_ROOT="$(docker info --format '{{.DockerRootDir}}')"
SR_AVAIL_KIB="$(df -Pk "$DOCKER_ROOT" | awk 'NR==2 {print $4}')"
if (( SR_AVAIL_KIB < 30 * 1024 * 1024 )); then
    printf 'Refusing: less than 30 GiB free under %s\n' "$DOCKER_ROOT" >&2
    exit 1
fi
# GPU 0 is also in the production service mask; the probe creates short-lived
# CUDA contexts on it. Require headroom so a concurrent production load is not squeezed.
SR_FREE_MIB="$(nvidia-smi --id="$GPU_UUID" --query-gpu=memory.free --format=csv,noheader,nounits)"
if (( SR_FREE_MIB < 2048 )); then
    printf 'Refusing: GPU %s has only %s MiB free\n' "$GPU_UUID" "$SR_FREE_MIB" >&2
    exit 1
fi

mkdir -p "$BASE"
OUT="$(mktemp -d "$BASE/cuda121_build_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
NAME="sr-$(basename "$OUT" | tr '[:upper:]_' '[:lower:]-')"
TAG="synergereader-test/ollama-cuda121:${NAME#sr-}"
printf 'Evidence: %s\nExperimental image: %s\n' "$OUT" "$TAG"
printf '%s\n' "$TAG" > "$OUT/image_tag.txt"
printf '%s\n' "$GPU_PCI" > "$OUT/gpu_pci.txt"
sha256sum "$0" > "$OUT/operator_script.sha256"

cleanup() {
    sr_exit_rc=$?
    trap - EXIT INT TERM HUP
    set +e
    if [ -s "$OUT/probe.cid" ]; then
        sr_cid="$(cat "$OUT/probe.cid")"
        if [[ "$sr_cid" =~ ^[0-9a-f]{64}$ ]] &&
           timeout 10s docker inspect "$sr_cid" >/dev/null 2>&1; then
            if ! timeout 20s docker rm -f "$sr_cid" > "$OUT/probe_cleanup.txt" 2>&1; then
                printf 'CLEANUP INCOMPLETE: probe container %s\n' "$sr_cid" >&2
                sr_exit_rc=1
            fi
        fi
    fi
    printf '\nEvidence retained: %s\n' "$OUT"
    exit "$sr_exit_rc"
}
trap cleanup EXIT
trap 'printf "BUILD/PROBE STOPPED at line %s\n" "$LINENO" >&2' ERR
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

printf '\nDownloading the two pinned source archives (about 63 MiB total)...\n'
curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
    --connect-timeout 15 --max-time 300 \
    "https://codeload.github.com/ollama/ollama/tar.gz/$OLLAMA_COMMIT" \
    -o "$OUT/ollama.tar.gz"
curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
    --connect-timeout 15 --max-time 300 \
    "https://codeload.github.com/ggml-org/llama.cpp/tar.gz/$LLAMA_COMMIT" \
    -o "$OUT/llama.cpp.tar.gz"
cat > "$OUT/SOURCE_SHA256SUMS.txt" <<'SR_HASHES'
afc434194f7e24f32bc0d59d74f5ce700d8745305e91cb751cca8746975b4752  ollama.tar.gz
ba920d45a2a916f2c5a100ef40cebbfd9deea866c553992a7f979e5344ba08de  llama.cpp.tar.gz
SR_HASHES
(cd "$OUT" && sha256sum -c SOURCE_SHA256SUMS.txt)

cat > "$OUT/Dockerfile" <<'SR_DOCKERFILE'
FROM nvidia/cuda@sha256:327c9e046fbf662275be0934742f7e5412f9b24402ee90bf4d649c1a21707912 AS builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates git python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir cmake==3.31.2 ninja==1.11.1.3
WORKDIR /src
COPY ollama.tar.gz llama.cpp.tar.gz SOURCE_SHA256SUMS.txt ./
RUN sha256sum -c SOURCE_SHA256SUMS.txt \
    && mkdir ollama llama.cpp \
    && tar -xzf ollama.tar.gz -C ollama --strip-components=1 \
    && tar -xzf llama.cpp.tar.gz -C llama.cpp --strip-components=1 \
    && test "$(cat ollama/LLAMA_CPP_VERSION)" = b10380
# 70-real emits native V100 machine code; it does not rely on newer PTX JIT.
# CUDA 12.1 predates the compressed fatbin requirement of the shipped 12.8 build.
# The official Ollama compatibility overlays are applied by its CMake project.
RUN cmake -S /src/ollama/llama/server -B /build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/dist \
        -DCMAKE_CUDA_ARCHITECTURES=70-real \
        -DCMAKE_CUDA_FLAGS='-t 1' \
        -DFETCHCONTENT_SOURCE_DIR_LLAMA_CPP=/src/llama.cpp \
        -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
        -DBUILD_SHARED_LIBS=ON \
        -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF \
        -DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=OFF -DGGML_CUDA_FA=OFF \
        -DGGML_CUDA_NCCL=OFF \
        -DOLLAMA_RUNNER_DIR=cuda_v12 -DOLLAMA_GPU_BACKEND=cuda \
    && cmake --build /build --target ggml-cuda --parallel 4 \
    && cmake --install /build --component llama-server --strip \
    && test -s /dist/lib/ollama/cuda_v12/libggml-cuda.so \
    && test -e /dist/lib/ollama/cuda_v12/libcudart.so.12 \
    && test -e /dist/lib/ollama/cuda_v12/libcublas.so.12 \
    && /usr/local/cuda/bin/cuobjdump --list-elf \
         /dist/lib/ollama/cuda_v12/libggml-cuda.so > /dist/cuda_images.txt \
    && grep -q sm_70 /dist/cuda_images.txt \
    && readelf -d /dist/lib/ollama/cuda_v12/libggml-cuda.so \
         | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p' > /dist/needed.txt \
    && ! grep -vxE 'libggml-base\.so\.0|libcudart\.so\.12|libcublas\.so\.12|libcublasLt\.so\.12|libcuda\.so\.1|libdl\.so\.2|librt\.so\.1|libpthread\.so\.0|libstdc\+\+\.so\.6|libm\.so\.6|libgcc_s\.so\.1|libc\.so\.6' /dist/needed.txt

FROM ollama/ollama@sha256:07e04ed05eb2232d66060e082da9ee6db9f37e9bf6d4260acb2535a6fd8e6ea7
USER root
# Replace this module and its CUDA libraries together; retain the original
# Ollama executable, llama-server and common libraries from the pinned image.
RUN rm -rf /usr/lib/ollama/cuda_v12
COPY --from=builder /dist/lib/ollama/cuda_v12/ /usr/lib/ollama/cuda_v12/
COPY --from=builder /dist/cuda_images.txt /opt/synergereader-cuda-images.txt
ENV OLLAMA_LLM_LIBRARY=cuda_v12
ENV OLLAMA_FLASH_ATTENTION=false
ENV OLLAMA_NO_CLOUD=1
LABEL org.synergereader.scope="experimental CUDA 12.1 / V100 SM70 compatibility trial" \
      org.synergereader.ollama-commit="e0e1d3c7d960f5879386871deb23582800c8599d" \
      org.synergereader.llama-commit="0b1bad14ff204627636aeb1de22ddcd5acb859d4"
# No GPU or model is needed for this link check. libcuda.so.1 is injected by the
# NVIDIA runtime at container start, so it is the only dependency allowed to be
# absent here; libggml-base.so.0 lives one directory up and is found via
# LD_LIBRARY_PATH at run time (Ollama sets it), not via $ORIGIN.
RUN LD_LIBRARY_PATH=/usr/lib/ollama:/usr/lib/ollama/cuda_v12 \
        ldd /usr/lib/ollama/cuda_v12/libggml-cuda.so > /opt/synergereader-cuda-ldd.txt \
    && ! grep 'not found' /opt/synergereader-cuda-ldd.txt \
         | grep -vq '^[[:space:]]*libcuda\.so\.1 => not found$' \
    && test "$(readlink /usr/lib/ollama/cuda_v12/libcudart.so.12)" = libcudart.so.12.1.105
SR_DOCKERFILE

# Send only the pinned sources and Dockerfile into the build, not host evidence.
printf '%s\n' '*' '!Dockerfile' '!ollama.tar.gz' '!llama.cpp.tar.gz' \
    '!SOURCE_SHA256SUMS.txt' > "$OUT/.dockerignore"
printf '\nBuilding the separate CUDA 12.1 module (builder download: about 3.7 GiB)...\n'
printf 'Compilation is limited to four jobs; no GPU is used during compilation.\n'
timeout --signal=TERM --kill-after=15s 5400s \
    docker build --progress=plain -t "$TAG" "$OUT" 2>&1 | tee "$OUT/build.log"
docker image inspect --format '{{.Id}}' "$TAG" | tee "$OUT/image_id.txt"

printf '\nChecking CUDA discovery for GPU 0; no model will be loaded...\n'
sr_probe_rc=0
timeout --signal=TERM --kill-after=10s 90s \
    docker run --rm --pull=never --cidfile "$OUT/probe.cid" \
    --runtime=nvidia --gpus "device=$GPU_UUID" \
    --network none --read-only --log-driver none \
    --tmpfs /tmp:size=256m,mode=1777 \
    --tmpfs /root/.ollama:size=16m,mode=0700 \
    --cap-drop ALL --security-opt no-new-privileges \
    --memory 4g --cpus 2 --pids-limit 256 \
    -e OLLAMA_DEBUG=2 -e OLLAMA_LLM_LIBRARY=cuda_v12 -e CUDA_MODULE_LOADING=EAGER \
    -e OLLAMA_NO_CLOUD=1 \
    --entrypoint /usr/bin/timeout "$TAG" \
    --signal=TERM --kill-after=5s 45s /bin/ollama serve \
    > "$OUT/gpu_discovery.log" 2>&1 || sr_probe_rc=$?
printf '%s\n' "$sr_probe_rc" > "$OUT/probe_exit_status.txt"

python3 -I -S -B - "$OUT" "$GPU_UUID" "$GPU_PCI" <<'SR_VERDICT'
import json
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
log = (root / "gpu_discovery.log").read_text(errors="replace")
rc = int((root / "probe_exit_status.txt").read_text())
compute_lines = [line for line in log.splitlines() if 'msg="inference compute"' in line]
gpu_lines = [line for line in compute_lines if re.search(r'\blibrary="?CUDA\b', line, re.I)]
bad_lines = [line for line in log.splitlines() if any(message in line.lower() for message in (
    "device kernel image is invalid", "nvidia driver too old", "failed to initialize cuda",
    "failed to load", "skipping cuda device",
))]
# Ollama logs pci_id=0000:BB:DD.F (CUDA format); nvidia-smi prints 00000000:BB:DD.F.
want_bdf = sys.argv[3].strip().lower().split(":", 1)[1]
pci_ok = False
if len(gpu_lines) == 1:
    m = re.search(r'\bpci_id="?([0-9A-Fa-f]+:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7])', gpu_lines[0])
    pci_ok = bool(m) and m.group(1).lower().split(":", 1)[1] == want_bdf
passed = (rc == 124 and len(compute_lines) == 1 and len(gpu_lines) == 1
          and pci_ok and "V100" in gpu_lines[0] and not bad_lines)
verdict = {
    "result": "GPU_DISCOVERY_PASS" if passed else "GPU_DISCOVERY_NOT_PASSED",
    "scope": "Startup discovery only; Qwen generation, GPU residency and reasoning are untested on this build.",
    "probe_exit": rc,
    "gpu_uuid": sys.argv[2],
    "gpu_pci": sys.argv[3],
    "pci_match": pci_ok,
    "compute_lines": compute_lines,
    "cuda_errors": bad_lines,
    "image": (root / "image_tag.txt").read_text().strip(),
    "image_id": (root / "image_id.txt").read_text().strip(),
}
(root / "RESULT.json").write_text(json.dumps(verdict, indent=2) + "\n")
print(json.dumps(verdict, indent=2))
if not passed:
    print("\nLast startup log lines:")
    print("\n".join(log.splitlines()[-90:]))
raise SystemExit(0 if passed else 3)
SR_VERDICT

printf '\nCUDA121_BUILD_AND_DISCOVERY_PASS\n'
printf 'Existing Qwen model volume was not mounted or changed.\n'
printf 'Next: one bounded generation request using the cached Qwen model.\n'
