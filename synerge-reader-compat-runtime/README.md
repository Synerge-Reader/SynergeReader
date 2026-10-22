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

## 5. Switching, and returning to Gemma or Llama when this runtime is down

The switch barrier waits for the previous model to leave by its own keep-alive. Which runtimes it
reads:

- **Always:** the primary runtime, and the previous and target profiles' runtimes.
- **This runtime too, only while `SYNERGE_EXPERIMENTAL_MODELS` is on.** With the tier off, nothing this
  backend loads can be resident here. A stopped or crashed compatibility container therefore never
  blocks a Gemma↔Llama switch, even when the URL is still set.
- **A persisted experimental selection** still reads its own runtime as the previous one.

Anything the barrier needs and cannot read aborts the switch. That covers:
- any status other than HTTP 200, including a redirect (redirects are never followed, and a redirect's
  body is never read as an inventory);
- a timeout or protocol error while the status line, headers or body are read;
- a body that is not JSON, or a malformed list;
- an ambiguous connection error, or an unreachable primary runtime.

The selection stays unchanged at a bumped epoch.

**One bounded exception, the return to control.** It applies when all of these hold:
- the switch is from an experimental profile on this runtime to Gemma or Llama on the primary runtime;
- the only runtime that cannot be read is this one, and either its URL is not configured on this server
  (for example after a restart without the URL), or a connection to it could **not be established**:
  - the connection was refused;
  - the host could not be reached or resolved;
  - the connect timed out.
- the window exceeds the previous profile's configured keep-alive.

A runtime that accepted the connection and then failed — including one that sent a status and part
of a body before stalling — does not qualify. Its response status is checked before the body is
read, and the response is always closed.

When they hold, the coordinator:
- keeps admission paused;
- requires all admitted work to drain, and no unknown or unresolved call (in memory and in the database);
- keeps polling every other runtime it needs;
- commits only after the **full 420 s residency window** that starts once draining has finished.

The window exceeds the previous profile's configured 5-minute keep-alive. That bounds how long this
backend's own last request should keep the model loaded, but whether the model actually expired and
unloaded is **not verified**.

The switch is then reported as **committed with `residency_unverified`** in the status and the
administrator panel, with a log line saying the timer does not prove the old model unloaded. The
database keeps `last_outcome = 'committed'`; there is no schema change.

It still fails closed when:
- this runtime answered but its list was unusable, or it was readable and still listed a blocking
  model at any point in the window;
- the primary runtime cannot be read, or lists a lingering registered model;
- work becomes uncertain;
- the window is shorter than 420 s or does not outlast the keep-alive;
- the switch is toward or between experimental profiles.

Throughout, an administrator can abort, and the commit keeps its compare-and-set fence. If this
runtime comes back with a readable, clean list during the window, the switch commits as an ordinary,
verified switch.

`recovery/restore_selection_to_gemma.sql` in the candidate bundle remains for the case the new
backend cannot run at all.

## 6. Default-off is not a pure no-op

With `SYNERGE_EXPERIMENTAL_MODELS` unset, these behaviours still change for every deployment of
this backend (release notes):

- **Heartbeats:** every `/ask` stream, Gemma's included, sends a `{"type":"heartbeat"}` line after 15 s of
  silence. Current and older frontends ignore unknown event types. `X-Accel-Buffering: no` and nginx's
  600 s read timeout are compatible with this.
- **Provenance v4:** every new answer record is schema version 4 (runtime key, experimental flag,
  think value). Versions 1–3 stay readable, and the admin history returns raw JSONB, so an older
  backend is not broken by v4 rows.
- **Switch drain timeout:** it rises from 480 s to 960 s for every switch. The coordinator settings
  take the longest job budget over all *registered* profiles, including disabled ones (Mistral's
  900 s + 60 s).
- **Registered experimental tags on the host Ollama block switches.** The barrier waits for every
  registered generation model, including experimental tags, on the runtimes it reads, even with the
  defaults. If another client loads `mistral-medium-3.5:128b` (the same digest `0341632a…` is in the
  host store) on the host Ollama, Gemma↔Llama switches wait for it and abort at the residency
  deadline. Operators should not run experimental tags on the host Ollama.
- **Completion lines:** each verifier and helper call, Gemma's included, logs one
  `[Generation] completion` line with counts only (role, profile, result, `done_reason`, token counts,
  thinking and reply lengths), never text.

## 7. Bounded candidate checks on the DGX

These are runtime-compatibility results only. They are not quality, capacity, vision or
citation-quality results. The Q&A-pair helper was **not exercised** in either run.

| Run | Snapshot / tree | Result |
| --- | --- | --- |
| `xm-20261020T030140Z-dc2e7a43` | tree `b44896f9…` | **FAIL, 20/22 gates.** Qwen and Mistral passed. Nemotron's answer completed (1,530 of 2,048 tokens), but its verifier and insights calls generated exactly their allowances, 512 and 2,048, without a usable reply. This led to Nemotron's own allowances and the completion diagnostics |
| `xm-20261020T192510Z-5a0f4ccc` (evidence archive SHA-256 `728b01ae…7482`) | tree `6a502336…` | **PASS.** Nemotron first: 14/14 gates. Then Qwen and Mistral: 18/18 gates |

Details of the passing run (`xm-20261020T192510Z-5a0f4ccc`):

| Profile | Answer prompt/generated tokens | Verifier (tokens of allowance) | Insights (tokens of allowance) | Residency | Answer total |
| --- | --- | --- | --- | --- | --- |
| Nemotron | 1,814 / 515 | 1,076 of 2,048, 1 claim judged | 4,088 of 4,096, stored | 100% (25.39 GiB) | 32 s |
| Qwen | 1,740 / 97 | 143 of 512 | 655 of 2,048 | 100% (22.31 GiB) | 80 s |
| Mistral | 2,338 / 185 | 302 of 512 | 601 of 2,048 | 83.62% (69.01 of 82.53 GiB) | 261 s, 16 heartbeats |

Gemma's D17 answer and the D12/T6R table regression check passed in both runs. Each switch took
about 300 s.

**Notes from the passing run:**
- **Nemotron's insights used 4,088 of its 4,096 tokens.** A slightly longer excerpt would be cut off; the
  result would then, safely, not be stored.
- **The compatibility runtime logged 10 GPU-discovery watchdog timeouts.** The cause is not
  established.

**What this evidence covers.** It validates tree `6a502336…` only. The later switch-coordinator
correction (the B2 runtime set and the B1 return to control) changes the tree, and has offline tests
but no DGX run.

## Limitations (recorded, not resolved)

- **Swap limit:** the DGX kernel has no swap-limit support. The daemon records `MemorySwap=-1`, so `memswap_limit` is not enforced; `mem_limit` is.
- **Mistral:** at most 91.27% GPU-resident in its smoke test (it uses CPU memory too). Its cold load was about 180 s. The backend gives it a 420 s per-read timeout within a 900 s job budget. A model switch can take several minutes.
- **Public cold-load survival:** answer streams send a heartbeat line after every 15 s of silence, so that proxies with an idle limit (~100 s on the public route) keep the stream open. The sender's behaviour is unit-tested; the private candidate uses loopback only. Whether a Mistral cold load survives the real public route (nginx and Cloudflare) is a later deployment check, not yet proved.
- **Switching on the shared GPU pool:** a switch waits until every registered generation model has left every runtime the barrier reads, by its own keep-alive expiry (see section 5 for which runtimes, and for the bounded return to control). The only exception is the target model on the runtime that will serve it. A copy of the target on the other runtime blocks too, and nothing is unloaded. Each switch took about 300 s in the DGX checks.
- **Context and output limits:** only 2,048-token contexts were smoke-tested. The application uses 8,192 tokens with conservative output allowances, thinking included: answer 2,048 for all three; verifier 512 and helper 2,048 for Qwen and Mistral; verifier 2,048 and helper 4,096 for Nemotron. Nemotron's are its own because in the first bounded candidate check (`xm-20261020T030140Z-dc2e7a43`) its verifier generated exactly 512 tokens and its insights helper exactly 2,048, and neither reply was usable. Thinking stays on for every role except Nemotron's document insights, which send `think: false` on role evidence for exactly that purpose and those settings (`synerge-reader-backend/diagnostics/NEMOTRON_INSIGHTS_THINK_FALSE_20261022.md`: one document, two calls); for its other roles, `think: false` and effort levels remain untested. A reply that reaches its allowance is still cut off: never a verdict, never stored, never retried. Each verifier and helper reply logs one `[Generation] completion` line with the role, profile, `done_reason`, token counts and the lengths (never the text) of its thinking and final reply. These are initial limits, not measured capacity.
- **Concurrency:** one combined answer-or-helper job at a time while an experimental profile is selected. Additional users get the existing bounded wait/busy behaviour. Not a measured capacity.
- **Not established:** quality, comparative performance, vision, citation quality and capacity are all deferred.
