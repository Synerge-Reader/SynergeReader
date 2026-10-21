"""The version-controlled compatibility-runtime recipe agrees with the profiles.

Reads ../synerge-reader-compat-runtime (PINS.json, the reviewed build recipe
and the compose overlay) as text. No Docker, model or network.

What these prove: the recipe file is byte-identical to the reviewed one
(SHA-256 b63b5c9a...); every pin the recipe hard-codes is the one PINS.json
records; the tags, manifest digests, think settings, runtime version and the
tested image ID in PINS.json are exactly the backend's experimental profiles;
and the overlay serves the runtime with the reviewed isolation and settings
(no published port, GPUs by UUID with GPU 2 excluded, read-only model store,
experimental switch off by default, the global OLLAMA_BASE_URL untouched).
"""

import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from generation_profiles import (  # noqa: E402
    COMPAT_RUNTIME,
    COMPAT_RUNTIME_OLLAMA_VERSION,
    COMPAT_RUNTIME_TESTED_IMAGE_ID,
    experimental_profiles,
)

RUNTIME_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "synerge-reader-compat-runtime")
REVIEWED_RECIPE_SHA256 = "b63b5c9a8edb92122ccfa31fb9158280a494587922ba4c05a89d3548bde3e351"


def _pins():
    with open(os.path.join(RUNTIME_DIR, "PINS.json"), encoding="utf-8") as fh:
        return json.load(fh)


def _text(*parts):
    with open(os.path.join(RUNTIME_DIR, *parts), "rb") as fh:
        return fh.read()


def test_the_build_recipe_is_the_reviewed_file_byte_for_byte():
    pins = _pins()
    data = _text(*pins["build"]["recipe"].split("/"))
    assert b"\r\n" not in data, "LF line endings (see .gitattributes); a CRLF checkout changes the hash"
    assert hashlib.sha256(data).hexdigest() == pins["build"]["recipe_sha256"] == REVIEWED_RECIPE_SHA256


def test_every_pin_the_recipe_hard_codes_is_recorded():
    build = _pins()["build"]
    recipe = _text(*build["recipe"].split("/")).decode("utf-8")
    assert f"RUNTIME='{build['runtime_image']}'" in recipe
    assert f"BUILDER='{build['builder_image']}'" in recipe
    assert f"OLLAMA_COMMIT={build['ollama_commit']}" in recipe
    assert f"LLAMA_COMMIT={build['llama_cpp_commit']}" in recipe
    assert f'= {build["llama_cpp_version"]}' in recipe
    for name, digest in build["source_archive_sha256"].items():
        assert f"{digest}  {name}" in recipe
    assert f"cmake=={build['build_tools']['cmake']}" in recipe and f"ninja=={build['build_tools']['ninja']}" in recipe
    assert f"-DCMAKE_CUDA_ARCHITECTURES={build['cuda_architectures']}" in recipe
    assert "-DGGML_CUDA_GRAPHS=OFF -DGGML_CUDA_FA=OFF" in recipe and "-DGGML_CUDA_NCCL=OFF" in recipe
    assert build["cuda_runtime_library"] in recipe


def test_pinned_models_are_exactly_the_experimental_profiles():
    pins = _pins()
    assert pins["runtime_key"] == COMPAT_RUNTIME
    assert pins["build"]["ollama_version"] == COMPAT_RUNTIME_OLLAMA_VERSION
    assert pins["tested_image"]["id"] == COMPAT_RUNTIME_TESTED_IMAGE_ID
    profiles = {p.profile_id: p for p in experimental_profiles()}
    assert {m["profile_id"] for m in pins["models"]} == set(profiles)
    for model in pins["models"]:
        profile = profiles[model["profile_id"]]
        assert (model["tag"], model["manifest_digest"]) == (profile.model_tag, profile.expected_digest)
        assert model["answer_think"] == profile.think and type(model["answer_think"]) is type(profile.think)
        assert model["source"] in pins["model_sources"]
        smoke = profile.runtime_smoke
        assert model["smoke"]["verdict"] == smoke.verdict and model["smoke"]["num_ctx"] == smoke.tested_num_ctx
        assert model["smoke"]["evidence"] in smoke.evidence_ref


def test_the_overlay_serves_the_runtime_with_the_reviewed_isolation():
    pins = _pins()
    overlay = _text("compose.compat-runtime.yml").decode("utf-8")
    service = overlay.split("  compat-ollama:", 1)[1]
    assert "ports:" not in service and "network_mode" not in service, "no published Ollama port"
    assert "internal: true" in overlay.split("services:", 1)[0]
    assert f"- {pins['serving']['service_alias']}" in service
    assert re.search(r"NVIDIA_VISIBLE_DEVICES=\$\{SYNERGE_COMPAT_GPU_UUIDS:\?", service)
    assert "compat_models:/cached:ro" in service and "read_only: true" in service
    for item in ("cap_drop:\n      - ALL", '"no-new-privileges:true"', "/tmp:size=2g", "/root/.ollama:size=64m",
                 "mem_limit: 128g", "cpus: 8", "pids_limit: 512", "driver: local", 'max-size: "10m"'):
        assert item in service, item
    for name, value in pins["serving"]["environment"].items():
        assert f"- {name}={value}\n" in service, name
    backend = overlay.split("  backend:", 1)[1].split("  compat-ollama:", 1)[0]
    assert "SYNERGE_COMPAT_OLLAMA_URL=http://synergereader-compat-ollama:11434" in backend
    assert "SYNERGE_EXPERIMENTAL_MODELS=${SYNERGE_EXPERIMENTAL_MODELS:-0}" in backend
    settings = [line for line in overlay.splitlines() if not line.lstrip().startswith("#")]
    assert not [line for line in settings if "OLLAMA_BASE_URL" in line], "the global endpoint is never repointed"
    assert pins["serving"]["gpu_indexes"] == [0, 1, 3] and pins["serving"]["gpu_indexes_never_used"] == [2]
