# Gemma 4 31B release qualification record (10 October 2026)

This record is the `evidence_ref` of `GEMMA_CANDIDATE_EVIDENCE` in
`generation_profiles.py`. It explains why `gemma_candidate` now records
`combined_capacity_verified=True` and `quality_verified=True`, which makes it
eligible in the shared admin model switcher.

The operator reviewed the evidence below and directed that both gates be
recorded. Being eligible only means an administrator may switch to Gemma
through the existing coordinated switch. It does not activate Gemma, change
the registry default (Llama) or change any persisted selection.

## 1. What changed and what did not

| Field | Before | After |
| --- | --- | --- |
| `think_false_verified` | `True` | `True` (unchanged) |
| `combined_capacity_verified` | `False` | `True` |
| `quality_verified` | `False` | `True` |
| `evidence_ref` | the earlier think:false smoke note | this record |
| `selectable` (derived) | `False` | `True` |

Unchanged: `requires_qualification=True` (every Gemma answer and helper job
still reads the installed digest first and refuses on a missing, unreadable or
different one), the pinned digest, `think=False`, the answer options
(temperature 0.2, `num_ctx` 8192, `num_predict` 1200), `keep_alive` "5m", the
150 s answer, verifier and helper read timeouts, the verifier options
(temperature 0.0, `num_predict` 16), the helper output limit (1200), the
application job deadlines, the Llama default and bootstrap, the admin-only
switch with draining and old-model residency checks, durable unresolved-call
records, no fallback, and the refusal of any per-request `model_profile_id`
(administrators included). Prompts, embeddings, retrieval, citation rules and
provenance are untouched.

## 2. Profile under qualification

| Item | Value |
| --- | --- |
| Profile id | `gemma_candidate` |
| Model tag | `gemma4:31b` (Q4_K_M, 31.3B parameters) |
| Pinned manifest digest | `6316f0629137b426c9d9b853ffc4c8209589f30ee39aebede6285096c0ff47e7` |
| Ollama runtime | 0.30.8 (observed in every run) |
| Embedder resident with it | `mxbai-embed-large:335m`, digest `468836162de7f81e041c43663fedbbba921dcea9b9fefea135685a39b2d83dd8`, dimension 1024 |
| Thinking policy | `think: false` sent; every answer reported 0 thinking characters |
| Control | `llama_control`, `llama3.1:8b`, digest `42182419e9508c30c4b1fe55015f06b65f4ca4b9e28a744be55008d21998a093` |

## 3. Evidence identity

### 3.1 Committed source the runs used

All runs used the harnesses committed at
`d0afc8852feb28a13b472fcec007bcbc4075bacc` from a `git archive` export. Every
value below was recomputed from the committed blobs and matches every run
record and summary. No mismatches were found.

| Item | SHA-256 |
| --- | --- |
| `main.py` (raw committed bytes) | `5f2118e81583edfa9b309c1db7bb5c00dfb25a8a3d5b90a0ca967551a10f2dea` |
| `_CITATION_LIMITS` | `3f858d7b886f41641ab2a88787f530302f2f1c2ecc64c3ea0d9b23f5a584887e` |
| `_CLAIM_VERIFIER_PROMPT` | `579ec4cee639f0254614194a9652627a73417e86796233a0ce9d0d4243769caf` |
| `_CLAIM_VERIFIER_WORDS` | `3331c7b3d6eccab17506982a67b141a1bfb2a11ebb2128bc8cb37cd63c4b51b6` |
| `_claim_verifier_status` | `66b5267411e70df0eaf264a9fb6337992988e5de1a47d1d94b7c0ba8d8e8169c` |
| `_ollama_claim_verifier` | `41972b1be724e5a161d193215499b6dba8d635547be6ea47b331b25b8849dc80` |

| Fixture | Revision | Fixture identity | Prompt | Gemma answer payload | Llama answer payload |
| --- | --- | --- | --- | --- | --- |
| C01 | 1 | `854a86bf9469b7b6cdfdae3088191355259a44f2b7779289c6edcbddb2c8cbe1` | `37a7901e17b4e806dfabea739fb54fd0b80480c32b28d1b756ba068f377aff49` | `9bbddd394fa57bf208645a2d0a8ef58005dd10c1a0a9460e8f8998255cecca47` | `4cbe4f5e8976a58648eff6a3288a26a546cd69efa5b1515e5b178eedb1673c05` |
| C02 | 1 | `e940fedf92b02c02089281bd13af1e92a9f9c5082529bfe73366c82076681a87` | `52be30a58be816974daebeab76bb01d8b306506c1acb8a8902faa35972881daf` | `712a684490aca6f35a5dee0d2da7b5366c46b4116275d67c2b70a5b2aea51325` | `980f92df69fa7834f727e07079abee2bb852011ec822fbcb324e45db7c248d83` |
| C03 | 1 | `f7fffb6c7165aa205bf3724d5314f71ab557c38f40e92e6622145c5f9a40d3bf` | `a858371fec22672380796728481d137ec70d994000dc2dab5750772fd49570e5` | `5d834661280758a32a5530bec0fc80d42e274b1088a3e3ee5bb64313ed96f711` | `78c66e8dcb7ed0eee55df0ec57df2d80984c44a048806914dd031834d5e50a78` |
| C04 | 2 | `6fd9f3172bb5323766aa101b619704f5c8aa0a6bff46bcf65ac31f227e05880e` | `492a1828a1297822c7054fb8614b6b8755aa9a95becc81407a0133a94d6cfec7` | `4cc910aabe41a0ef55501f5cad644b9d0dc96b6393feab211832202f72142350` | `168d3907d5eacd396e6141ab2e65594e7e491dbddff905605cbd8cf4ee5409a6` |
| C05 | 2 | `6b32291d9e313750e597a529f91f56a4cb2f4f2521772873c3db4130ca4f12b0` | `fc3e74cdc22ee70227f75e446dfdca260fba1fdfc05d2bda71a5122cf2383ed6` | `719e56ce5f630f829c6257ccec86753af24e2f7dc0afcead6b30a63c49b011ab` | `53a038877519019a2e7e2f67129cdcc32b77cb6b806c8ef428500cfc61b35cf1` |
| C06 | 1 | `9782405630a428bad072d444262351fb19dbd2481d65d97b0ef30c4a3ce05263` | `d231108846f4bc10b7d2bb2410dbf0139cbe348824118a889546aafe90518a55` | `67857441e3368c0f47d391716c481f68c59c2b46273eed4bd95ae7aebd3672f9` | `1beb737708dab7d0160e88eb4d97b646a27f5ef22a7988bcb7940fd24abde17f` |

The capacity probe's lanes used the same C04 and C05 fixture, prompt and Gemma
payload identities.

### 3.2 Evidence files

The files are kept unmodified in the operator's evidence store
(`SynergeReader_evidence/`). They are not in this repository. On 10 October
2026 the operator confirmed that the eight quality-file hashes match the
reviewed handoff. The handoff had a typo (an extra "e") in the Llama six-case
JSONL hash; the value below is the correct one.

| Run | File | SHA-256 |
| --- | --- | --- |
| `gemma_capacity_20261010_220826971` | `probe_gemma_capacity.jsonl` | `dac3d12c851e9238f382e5fca7105c05bab8dc3a1a78d74101e10de0e7f7d22e` |
| | `probe_gemma_capacity.jsonl.summary.json` | `5be5b9dcde66c5fb3c85cb46152e1ca6013ba59e4feba3570e9973cf4c1b4946` |
| | `gpu.csv` | `c3f8d05afa1d2863432a87ee3cf3103a570edca34665fdf7f483c1a92d85cb72` |
| | `preflight_version.json` | `f2101e3fc1526b2cff039d1e17337fb762ab0b800bfd916c260c12c8212827bb` |
| | `preflight_residency.json` | `698dc0b0aba1fe2d1ef5a356458cc6c96f3d98c34b867960fdd513a6dbdfd9f9` |
| `gemma_quality_20261010_221911413` | `gemma_c01_c06.jsonl` | `dbe4bed702a0e7004af4e92e3368ddde151df714b3685629416df6c87705b74c` |
| | `gemma_c01_c06.jsonl.summary.json` | `0fb2c273ada8fad0757cdbba2b926d311615a66f196042d858d55f6bfc5fb435` |
| | `preflight_version.json` | `f2101e3fc1526b2cff039d1e17337fb762ab0b800bfd916c260c12c8212827bb` |
| | `preflight_residency.json` | `b744e02575d38391754537c15c7c950757b42eba886fe3a1f580794e08029df7` |
| `gemma_repeat_20261010_225005371` | `gemma_repeat_c01_c05.jsonl` | `4814a9c88639b861da14bdb13c8b213e7be12a3d7910ebd8dcaaf368b991a96a` |
| | `gemma_repeat_c01_c05.jsonl.summary.json` | `059d9b585de2e8b92b96862bbcba981952eb0eab5191521c219e916d05c0f54c` |
| | `preflight_version.json` | `f2101e3fc1526b2cff039d1e17337fb762ab0b800bfd916c260c12c8212827bb` |
| | `preflight_residency.json` | `698dc0b0aba1fe2d1ef5a356458cc6c96f3d98c34b867960fdd513a6dbdfd9f9` |
| `llama_quality_20261010_225837813` | `llama_c01_c06.jsonl` | `1141451c79e61db0ccd06a4400fd4e5be4d8acabdc53fd3eff9b7711153d6513` |
| | `llama_c01_c06.jsonl.summary.json` | `0f70381ce9b5a2ea4a909a5884b9e0f4d8e42c28e8895e2e99a85eac500eda24` |
| | `preflight_version.json` | `f2101e3fc1526b2cff039d1e17337fb762ab0b800bfd916c260c12c8212827bb` |
| | `preflight_residency.json` | `698dc0b0aba1fe2d1ef5a356458cc6c96f3d98c34b867960fdd513a6dbdfd9f9` |
| `llama_repeat_20261010_230314040` | `llama_repeat_c01_c05.jsonl` | `bbb47aa36392de855b8c2de5c08387712c0d6c5501fc68f1d6fb9d9e8568b383` |
| | `llama_repeat_c01_c05.jsonl.summary.json` | `2b2f203a550e87253f82e5a9862a6a08e21db0b5e6f82eaf2e6d4dc9bd439b00` |
| | `preflight_version.json` | `f2101e3fc1526b2cff039d1e17337fb762ab0b800bfd916c260c12c8212827bb` |
| | `preflight_residency.json` | `d0bbca038da6077fb058b845bf3157ac90d19784718265685ce81e7bfa1745e1` |

| Run id | Run | Profile | Fixtures | Outcome |
| --- | --- | --- | --- | --- |
| `gcp-40002734529445dea22bc7c110a88d4d` | capacity | `gemma_candidate` | C04 and C05, overlapping | completed |
| `gq-eaec840f08924dd38218e87dedcbea1c` | quality | `gemma_candidate` | C01-C06 | completed |
| `gq-21958b73c99146198dc397fc406af74f` | repeat | `gemma_candidate` | C01, C05 | completed |
| `gq-0d6f1de739934be0bbc32ff99124b63b` | quality | `llama_control` | C01-C06 | completed |
| `gq-db4735babd264d02a1eedfd50ec4f55b` | repeat | `llama_control` | C01, C05 | completed |

Every run reports `identity.check: match`, Ollama 0.30.8, no stop category, no
unknown server execution and every worker reaped. Every answer call ended
`observed_terminal`, HTTP 200, `done_reason: stop`, accepted.

## 4. Combined capacity (probe `gcp-40002734529445dea22bc7c110a88d4d`)

- Identity matched for Gemma, the embedder and the runtime before any model
  call. One embedding (1024 dimensions, accepted), one smoke, then lanes A
  (C04) and B (C05) started together at 30.875 s.
- The two answers overlapped for 24.8 s, so peak in-flight generation calls
  were 2 (the planned maximum). 8 of 8 planned model calls ran, with at most 2
  verifier calls per answer. There were no missing observations and no
  unresolved execution.
- Each job (answer plus two verifier calls) took 37.6 s against the
  application's 420 s job budget. Warm Gemma load was about 1.2 s per call.
  The cold load at the smoke was 19.9 s.
- `/api/ps` during and after the workload: Gemma (`num_ctx` 8192, 25.9 GB) and
  the embedder both at 100 % VRAM, with the digests matching their pins.
- `gpu.csv` (232 samples, 17:07:53-17:12:38): Gemma's GPU peaked at 28,593 of
  32,768 MiB, leaving about 4.1 GiB free. The embedder's GPU peaked at 616 MiB.
  The other two GPUs stayed idle.

The probe's own summary keeps `combined_capacity_verified: false` and
`automatic_qualification: false` by design. A diagnostic never qualifies a
profile; recording the gate is this reviewed decision.

### 4.1 Against the proposed decision thresholds

The thresholds come from the qualification-preparation runbook (section 3.11).
That runbook names one quantity "worst-case job time" and computes it as
prefill time, plus 1200 divided by the output tokens per second, plus twice
the longest verifier call. The values below apply that formula to the
throughput observed on these short answers. They are **estimates** of a
full-length (1200-token) job. They are not worst-case bounds, and no
full-length answer was run or measured.

The Basis column separates the two kinds of result. **Measured** means
observed directly in the runs. **Estimate** means computed from short-answer
measurements.

| Quantity | Proposed threshold | Basis | Value | Result |
| --- | --- | --- | --- | --- |
| Gemma and embedder resident together at 100 % VRAM | Required | Measured | Yes, before, during and after the workload | Met |
| Two overlapping answer-plus-verifier jobs (C04, C05) | Within the 420 s application job budget | Measured | 37.6 s per job; 24.8 s answer overlap; peak 2 concurrent calls | Met |
| OOM or CUDA error lines | 0 | Not measured | No Ollama server log is in the evidence. All 8 calls returned HTTP 200, `done_reason: stop`, with no error, and both models stayed resident. | Not directly evidenced (indirect only) |
| Full-length job time, 2 lanes | 400 s or less | Estimate | 128.0 s (lane A: 19.8 s prefill, 13.3 tokens/s, 9.0 s verifier); 96.0 s (lane B) | Estimate within threshold; not measured |
| Full-length job time, 1 lane | 300 s or less | Estimate | 64.2 to 110.2 s across the 8 Gemma quality and repeat answers | Estimate within threshold; not measured |
| Warm first answer token (to be recorded against 5 s) | 5 s target | Measured | 5.5 to 7.3 s | Recorded: slower than the target by up to 2.3 s |
| Warm final completion of these short answers (to be recorded against 30 s) | 30 s target | Measured | 8.4 to 15.3 s | Within the target |
| Cold load, first Gemma call | 120 s maximum | Measured | 19.5 to 19.9 s at the smoke | Met |

Observed output speed on these short answers was 12.1 to 22.6 tokens/s on one
lane and 13.3 to 17.3 tokens/s on two. The 9 October window had seen 4.20 to
5.37 tokens/s on one lane, about 223 to 286 s for 1200 tokens alone. The
preparation runbook therefore expected the two-lane estimate to fail unless
throughput held under concurrency. In this window, two-lane short-answer
throughput gives an estimate within the threshold. That is still an estimate,
not a measured full-length pass.

## 5. Answer quality

The harness's original `manual_assessment` records, which say "pending" and
have null fields, are left unfilled in the evidence files. The findings below
are recorded here separately. They are Claude Code's source-based reading of
each answer against the frozen fixture excerpt, made on 10 October 2026, and
they are separate from the model verifier's labels (section 5.3) and the
heuristic anchors.

### 5.1 Gemma (quality run, with repeat for C01 and C05)

| Fixture | Answers every part | Citations appropriate | Unsupported statements | Notes |
| --- | --- | --- | --- | --- |
| C01 Step-Back gains (p.1) | Yes | Yes, `[C1]` | None | 7 %, 11 %, 27 % and 7 % are all correct for PaLM-2L. The repeat gave the same facts. |
| C02 Transformer BLEU and training (p.8) | Yes | Yes | None | 28.4 BLEU; 3.5 days on 8 P100 GPUs. |
| C03 Step-Back stages (p.3) | Yes | Yes | None | Both stages described faithfully, including "Abstraction-grounded Reasoning". The only uncited line is the lead-in "The two stages of Step-Back Prompting are:", which carries no fact. |
| C04 IRCoT operations and query (p.4) | Yes | Yes | None | Alternates retrieval and CoT generation; the last generated CoT sentence is the query. |
| C05 IRCoT termination and limits (p.4) | Yes | Yes | None | "answer is:" or the step limit; 8 steps and 15 paragraphs. The repeat gave the same facts. |
| C06 IRCoT vs Step-Back | Yes | Yes, `[C1]` IRCoT and `[C2]` Step-Back, correctly mapped | None | One minor paraphrase: "retrieve general principles", where the source retrieves facts about the principle. Retrieval sits in the right stage (with the abstraction question); only what is retrieved is loosened. |

Every factual sentence in Gemma's answers was cited. Apart from the C06
paraphrase noted above, I found no Gemma statement that departs from its
excerpt. This is one reader's assessment against short excerpts, not a
guarantee.

### 5.2 Llama control (same fixtures)

| Fixture | Finding |
| --- | --- |
| C01 | Correct and cited. The repeat was identical. |
| C02 | Correct and cited. |
| C03 | Incomplete: it names both stages but describes only Abstraction, in an uncited sentence that folds the reasoning step into it. |
| C04 | Correct, but the query sentence is uncited. |
| C05 | Correct and cited. The repeat was identical. |
| C06 | Stage imprecision. The Step-Back sentence assigns retrieval to Reasoning ("Reasoning, where facts are retrieved and used to reason about the solution"). The supplied Step-Back excerpt places retrieval in Abstraction ("ask a generic step-back question ... and retrieve relevant facts about the high-level concept or principle"); its Reasoning stage reasons from those facts. The IRCoT sentence is correct and cited. The opening and closing synthesis sentences are uncited. |

The C06 stage imprecision is recorded separately from the verifier's label:
the verifier marked that Step-Back claim `supported` (section 5.3).

Gemma met or exceeded the control on every fixture. Its C06 answer places
retrieval with the abstraction question, as the excerpt does.

Against the proposed assessment criteria from the qualification-preparation
plan: there is no regression against Llama on C01-C03 (Gemma is better on
C03); C04 gives the last-generated-sentence query; C05 gives the "answer is"
and step-limit rule with 8 steps and 15 paragraphs; C06 gives both mechanisms
with both sources cited correctly; and there is no fabricated quote and no
critical unsupported claim. All are met.

### 5.3 Model verifier labels and heuristic anchors (not truth)

- Gemma: 11 verifier calls; all 11 cited claims labelled `supported`. One
  uncited line (the C03 lead-in) was `unsupported/no_citation` by the
  deterministic citation check. Heuristic anchors hit every group in all six
  fixtures. This is self-verification: Gemma judged its own answers.
- Llama: 8 cited claims `supported`; 4 uncited sentences `no_citation`. One
  of the `supported` labels is on the C06 Step-Back claim, which section 5.2
  finds imprecise about which stage retrieves. The label and the finding
  disagree, which is one reason verifier labels are not treated as truth.

### 5.4 Latency observed (short answers)

Gemma answer time: C01 41.6 s, which includes a 23.1 s reload from the smoke's
`num_ctx` 4096 to the answer's 8192; then 8.4 to 15.3 s for the others
(quality and repeat runs). Gemma verifier calls took 2.6 to 7.2 s. Gemma
answers used 45 to 97 output tokens against the 1200 limit. Llama answers took
1.7 to 12.6 s and used 38 to 139 tokens.

## 6. Why Gemma is now eligible

1. **Identity:** the pinned digest, runtime and embedder matched in every run,
   and every Gemma job keeps checking its identity before any model call.
2. **Capacity:** Gemma and the embedder stayed jointly resident. Two
   overlapping answer-plus-verifier jobs finished at about 9 % of the job
   budget, with GPU headroom left (measured). At the observed short-answer
   throughput, the runbook formula estimates 128 s for a full 1200-token job
   on two lanes, within the proposed 400 s threshold. That is an estimate, not
   a worst-case bound or a measured full-length pass.
3. **Quality:** on six frozen fixtures from three public papers, Gemma
   answered every part with cited statements consistent with the excerpts in
   this reading (one minor paraphrase in C06). It did at least as well as the
   production Llama control. The C01 and C05 repeats were consistent.
4. **Thinking:** `think:false` held in every call (0 thinking characters).

## 7. Limitations

- Six short, frozen-evidence fixtures from three public papers: Step-Back
  Prompting, Attention Is All You Need, and IRCoT. They do not
  exercise retrieval, uploaded or private documents, legal or medical
  material, multi-turn chat or website latency. L01 is not represented, and
  C06 is a controlled two-paper scope.
- Gemma's answers were short (at most 97 output tokens). Latency and memory
  for full-length answers near the 1200-token limit were not measured.
- Capacity covered two concurrent short jobs only. It did not cover uploads or
  embedding throughput, helper jobs, the application's admission ledger, or
  more than two answers. `/api/ps` and `nvidia-smi` are point-in-time views
  and do not prove exclusive GPU use.
- No Ollama server log was collected, so "zero OOM or CUDA error lines" rests
  only on the clean call outcomes and continuous residency.
- The full-length job times (64.2 to 128.0 s) are estimates at observed
  short-answer throughput. They are not worst-case bounds or measured
  full-length passes. Throughput may fall as a long answer fills the context
  window, and verifier calls may take longer on longer claims.
- Warm first-token latency (5.5 to 7.3 s) is slower than the proposed 5 s
  target.
- Cold loads vary: 91.8 s in the earlier smoke and 19.5 to 23.4 s here. The
  150 s per-read timeouts are retained.
- Verifier labels are self-verification and are not independent evidence.
- Repeats were n=2 for C01 and C05 only. Sampling variance at temperature 0.2
  is otherwise uncharacterized.
- The historical dry-run report hashes were produced before release. Dry runs
  after this change differ only in their reported qualification fields.
