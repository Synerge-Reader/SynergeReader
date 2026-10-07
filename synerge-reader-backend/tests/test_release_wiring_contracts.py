"""Static contracts for how the release commit reaches /version.

Source-only: parses docker-compose.yml, the deployment workflow, the frontend
Dockerfile and frontend source as files. Nothing is run -- no docker-compose
(not even ``config``, which would interpolate a real local .env), no workflow,
no container, network, database, or subprocess.

The chain these pin: the deployment workflow binds the commit actions/checkout
checked out (``${{ github.sha }}``) to SYNERGE_BUILD_SHA for the step that
builds and starts the containers; docker-compose passes that variable through
to the backend container, bare, so an unset value stays absent; the backend
reads exactly that name and treats a missing or empty value as unknown; and
the frontend needs no build-time commit because it displays /version.
"""

from pathlib import Path
import os
import re
import sys

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from release_metadata import BUILD_SHA_ENV, release_metadata_from_env

_REPO = Path(__file__).resolve().parents[2]
_COMPOSE = _REPO / "docker-compose.yml"
_WORKFLOW = _REPO / ".github" / "workflows" / "push_to_server.yml"
_FRONTEND = _REPO / "synerge-reader-frontend"

_DEPLOY_STEP = "Build & Start Containers"
_FULL_SHA = re.compile(r"(?<![0-9A-Fa-f])[0-9A-Fa-f]{40}(?![0-9A-Fa-f])")


@pytest.fixture(scope="module")
def compose():
    return yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def workflow_steps():
    workflow = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    return workflow["jobs"]["deploy"]["steps"]


def test_the_backend_reads_the_name_the_deployment_supplies():
    assert BUILD_SHA_ENV == "SYNERGE_BUILD_SHA"


def test_compose_forwards_the_build_sha_to_the_backend_only(compose):
    services = compose["services"]
    backend_environment = services["backend"]["environment"]
    assert BUILD_SHA_ENV in backend_environment, (
        "a bare pass-through: supplied values reach the backend, an unset one stays absent"
    )
    assert not [entry for entry in backend_environment if str(entry).startswith(f"{BUILD_SHA_ENV}=")], (
        "no value, default, or interpolation is fixed in the compose file"
    )
    for name, service in services.items():
        if name != "backend":
            assert BUILD_SHA_ENV not in str(service), f"{name} must not receive the build SHA"


def test_the_deployment_step_binds_the_checked_out_commit(workflow_steps):
    names = [step.get("name") for step in workflow_steps]
    [deploy] = [step for step in workflow_steps if step.get("name") == _DEPLOY_STEP]
    assert deploy["env"] == {BUILD_SHA_ENV: "${{ github.sha }}"}
    # Step env applies to the whole script, so both commands see it.
    assert "docker-compose build" in deploy["run"] and "docker-compose up" in deploy["run"]

    [checkout] = [step for step in workflow_steps if str(step.get("uses", "")).startswith("actions/checkout@")]
    assert names.index(checkout.get("name")) < names.index(_DEPLOY_STEP)
    assert "ref" not in (checkout.get("with") or {}), (
        "without a ref override, actions/checkout checks out github.sha itself"
    )


@pytest.mark.parametrize("path", [_COMPOSE, _WORKFLOW], ids=["docker-compose", "workflow"])
def test_no_commit_sha_is_hardcoded_in_the_release_wiring(path):
    assert not _FULL_SHA.findall(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("environ", [{}, {BUILD_SHA_ENV: ""}], ids=["absent", "empty"])
def test_an_unsupplied_sha_is_still_unknown(environ):
    release = release_metadata_from_env(environ)
    assert release.build_known is False and release.build_sha is None
    assert release.display_version == "v0.1.0 · build unknown"


def test_the_frontend_needs_no_build_time_commit(compose):
    frontend_args = compose["services"]["frontend"]["build"].get("args") or []
    dockerfile = (_FRONTEND / "Dockerfile").read_text(encoding="utf-8")
    source_vars = set()
    for path in (_FRONTEND / "src").rglob("*"):
        # Shipped source only. VersionBadge.test.jsx sets decoy commit
        # variables on purpose to prove the badge ignores them.
        if path.suffix in {".js", ".jsx"} and not path.name.endswith((".test.js", ".test.jsx")):
            source_vars.update(re.findall(r"process\.env\.([A-Z0-9_]+)", path.read_text(encoding="utf-8")))
    assert "REACT_APP_BACKEND_URL" in source_vars, "the scan really read the application source"

    commit_like = re.compile(r"SHA|COMMIT|GIT|BUILD_ID|VERSION", re.IGNORECASE)
    for name in [str(arg).split("=", 1)[0] for arg in frontend_args] + re.findall(r"^ARG\s+(\S+)", dockerfile, re.M):
        assert not commit_like.search(name), name
    assert not [name for name in source_vars if commit_like.search(name)], (
        "the frontend shows the server's /version answer, never its own build identity"
    )
