# Compatibility generation runtime (`compat_cuda121`)

A second, private Ollama runtime for the three **experimental** reasoning models:

| Profile | Model | Manifest digest | Answer think | Runtime smoke (operator-supplied, DGX) |
| --- | --- | --- | --- | --- |
| `qwen38_27b_experimental` | `qwen3.8:27b` | `aaee06c3…e509097813` | `true` | PASS: 100% GPU (20.42 GiB), 11.69 s incl. 9.28 s load |
| `nemotron35_lightning_30b_experimental` | `nemotron-3.5-lightning:30b` | `e6905fff…e2f17f4dc` | `true` | PASS: 100% GPU (23.65 GiB), 16.65 s incl. 12.13 s load |
| `mistral_medium35_128b_experimental` | `mistral-medium-3.5:128b` | `0341632a…b7c7b1401e9d70df` | `"high"` | REVIEW_REQUIRED: only the original 99% GPU-residency target failed (91.27%, 70.35 of 77.08 GiB); 200.72 s incl. 180.38 s load |

Each smoke test was **one synthetic invoice calculation at a 2,048-token context**. These are
observations from single requests on the DGX, supplied by the operator. They are not
a quality baseline, a comparative benchmark, a vision test, a citation-quality result
or a capacity qualification, and none of those is claimed here. The application's
own unit tests do not reproduce any DGX inference.

Gemma, Llama and the configured embedder stay on the existing host Ollama. The global
`OLLAMA_BASE_URL` is never pointed at this runtime.

## Files

| Path | What it is |
| --- | --- |
| `build/SynergeReader_Qwen38_CUDA121_Build_20261019_reviewed.sh` | The **reviewed** build recipe, byte-for-byte (SHA-256 `b63b5c9a8edb92122ccfa31fb9158280a494587922ba4c05a89d3548bde3e351`). `.gitattributes` keeps it LF on every checkout. |
| `PINS.json` | Every pin: base runtime and builder images by digest, Ollama/llama.cpp commits, source archive hashes, build flags, the tested image ID and tag, the serving settings, the model sources and the three model identities. `synerge-reader-backend/tests/test_compat_runtime_pins.py` checks it against the recipe, the overlay and the backend's profiles. |
| `stage_model_store.py` | Builds the dedicated model store from the existing caches (pinned manifests and their blobs only). Standard library only. Tests: `tests/`. |
| `compose.compat-runtime.yml` | The serving overlay: the `compat-ollama` service and the backend's two extra settings. |

## 1. The image

The recipe keeps the official Ollama 0.32.12 image (`ollama/ollama@sha256:07e04ed0…`) and
rebuilds only its matching CUDA module with CUDA 12.1.1, native SM70 (`70-real`), Flash
Attention off, CUDA graphs off and NCCL off. Its evidence directory keeps the generated
Dockerfile, the source checksums, the build log, the image ID and a GPU-discovery verdict.

The first candidate **reuses the tested local image by ID**:
`sha256:96e3cc4e4155ef96d11578415f1ac36892d8bdc3a83c40623cc8ffdfbf4392ed`
(tag `synergereader-test/ollama-cuda121:cuda121-build-20261019t155454z-2sxha4`, build evidence
`cuda121_build_20261019T155454Z_2Sxha4`). The backend records this ID in each profile's smoke
evidence.

A rebuild may produce a different image ID. Record it, run the recipe's discovery check and
a bounded compatibility check on the new image, and only then substitute it, together with a
reviewed change to `COMPAT_RUNTIME_TESTED_IMAGE_ID`.

## 2. The model store

One dedicated Docker volume, `synergereader-compat-models-20261019`, populated from the
existing caches. Nothing is downloaded, pruned, modified or deleted in any source:

| Model | Source | Ollama models directory inside it |
| --- | --- | --- |
| Qwen | volume `sr-qwen38-20261019T032515Z-x17f6I-models` | `models/` |
| Nemotron | volume `sr-nemotron35-lightning-20261019-models` | `models/` |
| Mistral | host directory `/usr/share/ollama/.ollama/models` | (the directory itself) |

`stage_model_store.py` copies only the three pinned manifests and the blobs they reference
(config and every layer). It verifies each manifest's SHA-256 against the pin and each
blob's size and SHA-256 while copying. It re-hashes the staged copy and publishes it with
one rename to `models/` in the volume, so a partial store is never visible. It refuses
before copying when free space is below the requirement plus headroom (default 16 GiB),
when a store is already published, or when another run holds the lock. Expect roughly
115 GiB. **Run `plan` first:** it measures without writing.

Run it inside an existing local image that has `python3` (the candidate uses its backend
image, by ID). Run it with no network and a read-only root, with every source mounted
read-only. `DAC_READ_SEARCH` is the only capability kept: it lets root read the sources
regardless of their owners.

```bash
STAGER_IMAGE=sha256:...   # a local image with python3, by ID
docker volume create --label synergereader.purpose=compat-model-store synergereader-compat-models-20261019
for CMD in plan stage; do
docker run --rm --pull=never --network none --read-only --user 0:0 \
  --cap-drop ALL --cap-add DAC_READ_SEARCH --security-opt no-new-privileges \
  --memory 4g --cpus 2 --pids-limit 64 --log-driver none \
  --mount "type=bind,src=$PWD/synerge-reader-compat-runtime,dst=/tool,readonly" \
  --mount "type=volume,src=sr-qwen38-20261019T032515Z-x17f6I-models,dst=/src/qwen,readonly,volume-nocopy" \
  --mount "type=volume,src=sr-nemotron35-lightning-20261019-models,dst=/src/nemotron,readonly,volume-nocopy" \
  --mount "type=bind,src=/usr/share/ollama/.ollama/models,dst=/src/mistral,readonly" \
  --mount "type=volume,src=synergereader-compat-models-20261019,dst=/dest,volume-nocopy" \
  --entrypoint python3 "$STAGER_IMAGE" -I -B /tool/stage_model_store.py "$CMD" --pins /tool/PINS.json \
  --source qwen=/src/qwen/models --source nemotron=/src/nemotron/models --source mistral=/src/mistral \
  --dest /dest || break
done
```

`verify --store /dest/models` (with the volume mounted read-only) re-checks a published
store. Add `--quick` to check sizes only.

## 3. Serving

Add `compose.compat-runtime.yml` after the release compose file. It was checked with
docker-compose 1.29.2 (`config`) merged onto the live-extraction release file: the backend
keeps `default` and `extract_private` and gains `compat_private`. The render step must supply
four verified values:

| Variable | Value |
| --- | --- |
| `SYNERGE_COMPAT_IMAGE_ID` | the image ID above, after `docker image inspect` confirms it |
| `SYNERGE_COMPAT_GPU_UUIDS` | `nvidia-smi --id=<i> --query-gpu=uuid` for indexes **0, 1 and 3**, comma-separated; GPU 2 is never listed |
| `SYNERGE_COMPAT_MODEL_VOLUME` | `synergereader-compat-models-20261019`, after `verify` passes |
| `SYNERGE_EXPERIMENTAL_MODELS` | `0` (default) until a release enables the experimental tier; `1` in the candidate |

The service has no published port. It sits only on the internal network `compat_private` and
answers there to the alias `synergereader-compat-ollama`. Because the backend also joins the
shared external network, the render step must prove that alias is unique. No container on
any network the backend joins may answer to that name, and from inside the backend the name
must resolve only to the compat container's `compat_private` address.

Other settings:
- **Container hardening:** read-only root; `/tmp` (2 GiB) and `/root/.ollama` (64 MiB) are tmpfs; all capabilities dropped; no-new-privileges.
- **Limits:** 128 GiB memory, 8 CPUs, 512 pids, `local` logs (10 MiB × 2). These are the limits Mistral was smoke-tested with at 2,048 tokens. The bounded candidate check records whether they hold at the application's 8,192.
- **Ollama:** `OLLAMA_NOPRUNE=1`, `OLLAMA_NO_CLOUD=1`, `OLLAMA_LLM_LIBRARY=cuda_v12`, Flash Attention off, `OLLAMA_KV_CACHE_TYPE=f16`, `OLLAMA_SCHED_SPREAD=1` (the tested Mistral setting), `OLLAMA_NUM_PARALLEL=1`, `OLLAMA_MAX_LOADED_MODELS=1`.
- **Context and keep-alive:** `OLLAMA_CONTEXT_LENGTH=8192`, matching every role call the application sends, so roles never reload a model. A 5-minute keep-alive.

## 4. Backend settings

| Setting | Effect |
| --- | --- |
| `SYNERGE_COMPAT_OLLAMA_URL` | `http://host:port` only; no credentials, path, query or HTTPS. Malformed or unset: the runtime is not configured, the three profiles are refused with a fixed message, and Gemma/Llama are unaffected. The value is never logged or returned. |
| `SYNERGE_EXPERIMENTAL_MODELS` | `1`/`true`/`yes` enables the experimental tier; anything else (the default) leaves the three profiles visible to the administrator but not selectable. |

A persisted experimental selection survives a restart unchanged. If the tier is off or the
runtime unconfigured, answers are refused with a 503 and the administrator panel says why.
Nothing switches automatically.

## Limitations (recorded, not resolved)

- **Swap limit:** the DGX kernel has no swap-limit support. The daemon records `MemorySwap=-1`, so `memswap_limit` is not enforced; `mem_limit` is.
- **Mistral:** at most 91.27% GPU-resident in its smoke test (it uses CPU memory too). Its cold load was about 180 s. The backend gives it a 420 s per-read timeout within a 900 s job budget. A model switch can take several minutes.
- **Public cold-load survival:** answer streams send a heartbeat line after every 15 s of silence, so that proxies with an idle limit (~100 s on the public route) keep the stream open. The sender's behaviour is unit-tested; the private candidate uses loopback only. Whether a Mistral cold load survives the real public route (nginx and Cloudflare) is a later deployment check, not yet proved.
- **Switching on the shared GPU pool:** a switch waits until every registered generation model has left every runtime by its own keep-alive expiry. The only exception is the target model on the runtime that will serve it. A copy of the target on the other runtime blocks too, and nothing is unloaded.
- **Context and output limits:** only 2,048-token contexts were smoke-tested. The application uses 8,192 tokens with conservative output allowances, thinking included: answer 2,048 for all three; verifier 512 and helper 2,048 for Qwen and Mistral; verifier 2,048 and helper 4,096 for Nemotron. Nemotron's are its own because in the first bounded candidate check (`xm-20261020T030140Z-dc2e7a43`) its verifier generated exactly 512 tokens and its insights helper exactly 2,048, and neither reply was usable. Thinking stays on for every role: the runtime accepts `think: false` and effort levels for Nemotron, but neither has been tested with it. A reply that reaches its allowance is still cut off: never a verdict, never stored, never retried. Each verifier and helper reply logs one `[Generation] completion` line with the role, profile, `done_reason`, token counts and the lengths (never the text) of its thinking and final reply. These are initial limits, not measured capacity.
- **Concurrency:** one combined answer-or-helper job at a time while an experimental profile is selected. Additional users get the existing bounded wait/busy behaviour. Not a measured capacity.
- **Not established:** quality, comparative performance, vision, citation quality and capacity are all deferred.
