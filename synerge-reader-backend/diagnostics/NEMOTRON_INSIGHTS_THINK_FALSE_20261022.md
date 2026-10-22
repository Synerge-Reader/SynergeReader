# Nemotron document insights: `think: false` evidence (22 October 2026)

This record is the `evidence_ref` of `NEMOTRON35_INSIGHTS_THINK_FALSE_EVIDENCE`
in `generation_profiles.py`. It is the only basis on which the experimental
profile `nemotron35_lightning_30b_experimental` sends `think: false`. That
applies to its document-insights helper calls only.

## Why

Production acceptance S13 for PR #58 passed 35 of 36 gates. The failed gate is
Nemotron's insights helper. With thinking on (temperature 0.1, `num_predict`
4096) it spent its whole allowance thinking: `done_reason=length`, 13,431
thinking characters and an empty reply. Nothing was stored, as designed.

## The run

| Item | Value |
| --- | --- |
| Archive | `pr58-prod-nemotron-diag-20261022T071320Z.tar.gz` |
| Archive SHA-256 | `7246da7d235f18e17e37a0b89c993d10bbf315a794055e3bc7f54780fe35f3f8` |
| Tool | Nemotron insights diagnostic rev2, hash `8de4a07d5befb16a` |
| Release | `fc985428324ec252cdad303733f0d14ac9169457` |
| Run status and decision | VALID, THINK_FALSE (decision rule fixed before the run) |
| Model | `nemotron-3.5-lightning:30b`, digest `e6905fffe2e2d9bbe325a9a2d2e074e3bd32683721bc6dfba409dede2f17f4dc` (matched the pin) |
| Runtime | Ollama 0.32.12, image `sha256:96e3cc4e4155ef96d11578415f1ac36892d8bdc3a83c40623cc8ffdfbf4392ed` |
| Isolation | a private runtime copy on GPUs 0, 1 and 3 (GPU 2 excluded), read-only model store, internal network only; removed afterwards, and ingress reopened |
| Residency | Nemotron alone, fully in GPU memory, context 8,192 tokens |
| Input | the deployed insights prompt (1,218 prompt tokens) on one document of 2,379 characters: the same prompt as the failed S13 call |

## Results

There were six calls, two per setting. All six were valid and stopped
naturally (`done_reason=stop`).

The headroom criterion, fixed before the run, required a usable reply in at
most 75% of the 4,096-token allowance (3,072 tokens).

| Calls | `think` | Temperature | Output tokens | Facts | Headroom | Duration |
| --- | --- | --- | --- | --- | --- | --- |
| A1, A2 | true | 0.1 | 3,454 and 3,942 | 5 and 5 | not met | 46.6 s (cold load) and 38.1 s |
| B1, B2 | true | 1.0 | 3,693 and 3,487 | 5 and 5 | not met | 35.1 s and 33.2 s |
| **C1** | **false** | 0.1 | **336** | 6 | met | **3.979 s** |
| **C2** | **false** | 0.1 | **332** | 6 | met | **4.066 s** |

Both C calls:
- honoured `think: false`: no thinking field and no inline think marker;
- returned usable JSON with six facts, ten keywords and eight entities;
- sent exactly `{"model": "nemotron-3.5-lightning:30b", "think": false,
  "stream": false, "keep_alive": "5m", "options": {"temperature": 0.1,
  "num_ctx": 8192, "num_predict": 4096}}`, with a 300 s read timeout.

These are the application's existing insights settings with thinking turned
off.

## What the code does with it

- **Insights call:** Nemotron's insights calls send `think: false` with
  temperature 0.1, `num_ctx` 8192, `num_predict` 4096 and the existing 300 s
  helper read timeout. Job budgets are unchanged.
- **When it applies:** the override is accepted only with `RoleThinkEvidence`
  that matches all of the following:
  - the profile and the purpose (`insights`);
  - the model tag and digest;
  - the runtime key, version and image;
  - the context, output limit and temperature;
  - the identity the job observed.
- **When it does not:** missing or mismatched evidence refuses the call before
  dispatch. The call never falls back to another think setting.
- **Other roles and profiles:** Nemotron's answer, verifier and
  knowledge-base Q&A helper keep thinking on with their existing limits.
  Every other profile is unchanged.
- **Completion and storage:** strict completion, JSON parsing and storage are
  unchanged. A truncated, empty or unusable reply is never stored.
- **Tier:** Nemotron stays experimental. This evidence is not a
  `Qualification`, and Nemotron's runtime-smoke record is unchanged.

## Limits

- One document, two calls per setting. The run says nothing about insight
  quality in general, other documents, capacity, latency under load or
  production acceptance.
- The knowledge-base Q&A prompt was not exercised, so that helper keeps
  thinking on.
- Offline tests check the request the backend builds; they do not reproduce
  inference.

## Status

- Production S13 remains **FAILED at 35/36** until this reviewed change is
  deployed and acceptance is rerun.
- The acceptance checkpoints and criteria are unchanged.
