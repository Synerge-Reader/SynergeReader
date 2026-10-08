"""Real startup DDL and /ask answer persistence against a disposable PostgreSQL.

Opt-in integration layer, like tests/test_migrate_embeddings_script.py and
tests/test_chat_history_provenance_migration.py: every test here skips unless
SYNERGE_MIGRATION_TEST_DSN names a throw-away PostgreSQL database with the
pgvector extension installed in ``public``. Example:

    SYNERGE_MIGRATION_TEST_DSN=postgresql://user@localhost:5432/somedb \\
        python -m pytest tests/test_startup_and_answer_persistence_integration.py

Nothing runs at collection time beyond importing pure modules: no fixture,
connection, startup code or route compilation.

Each test works inside its own freshly created schema (``app_test_<hex>``),
first on the search_path, so every unqualified application table that
``init_db`` creates or the /ask route writes resolves there. ``public`` is kept
on the path only for the pgvector type; it holds no application table. The
schema is dropped afterwards with everything in it.

What is REAL here:
* ``dbSetup.init_db`` itself -- its DDL, schema validation, single commit, and
  its rollback/close error handling -- unmodified.
* The current /ask route: ``ask_question`` and the definitions it calls in
  main.py (model resolution, citation resolution, claim verifier) are compiled
  from the current main.py source, the same AST approach as
  tests/test_main_citation_wiring.py, and mounted on an isolated FastAPI app
  with no lifespan, driven in-process by TestClient (no listener).
* The completed-stream reader, answer provenance construction and JSON
  encoding, NDJSON events, the pinned default model registry, release
  metadata and the configured embedding profile (pure value objects built from
  an explicit empty configuration), and the route's actual INSERT, executed by
  a real psycopg2 connection and cursor.

What is MOCKED, and therefore NOT qualified here:
* ``connect_to_postgres``: replaced by a connector to the exact DSN that sets
  the case schema's search_path. This intentionally does not exercise the
  production connector's dotenv/DB_CONNECTION_STRING configuration.
* Authentication (``_resolve_caller_id`` returns a seeded synthetic user),
  evidence planning (a synthetic one-passage bundle), and the Ollama transport
  (a synthetic completed stream and verifier reply). No model, embedder,
  network service or application server is contacted.

So this is startup-DDL and persistence integration coverage on a fresh
disposable target. It does not qualify application lifespan, production
configuration, model quality, real embeddings, other pre-existing schema
states, lock contention, or production PostgreSQL version parity.
"""

import ast
import datetime
import json
import os
from pathlib import Path
import re
import sys
from typing import Optional
import uuid

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_MAIN_PATH = Path(__file__).resolve().parents[1] / "main.py"

_DSN = os.environ.get("SYNERGE_MIGRATION_TEST_DSN")
needs_db = pytest.mark.skipif(not _DSN, reason="set SYNERGE_MIGRATION_TEST_DSN to run the database tests")

_APPLICATION_TABLES = {
    "users", "documents", "document_chunks", "chat_history",
    "knowledge_base", "admin_audit_log", "document_insights",
}
_SYNTHETIC_TOKEN = "synthetic-integration-token"
_QUESTION = "How long is the synthetic agreement term?"
_ANSWER_TOKENS = ("The synthetic agreement term is ", "two years [C1].")
_EVIDENCE_TEXT = "SYNTHETIC AGREEMENT. The agreement term is two years from signature."
_INJECTED_FAILURE = "synthetic late init_db failure"
_INSERT_FAILURE = "synthetic chat_history insert failure"


# --- the disposable target ------------------------------------------------------


class _RecordingConnection:
    """Delegates everything to one real psycopg2 connection and records
    commit/rollback/close in the shared log. With ``fail_statement`` set, its
    cursor replaces that one statement with a real PostgreSQL error."""

    def __init__(self, real, index, log, fail_statement=None):
        self._real = real
        self._index = index
        self._log = log
        self._fail_statement = fail_statement

    @property
    def real(self):
        return self._real

    def cursor(self, *args, **kwargs):
        cursor = self._real.cursor(*args, **kwargs)
        if self._fail_statement is None:
            return cursor
        return _FailingCursor(cursor, self._index, self._log, self._fail_statement)

    def commit(self):
        self._log.append(("commit", self._index))
        return self._real.commit()

    def rollback(self):
        self._log.append(("rollback", self._index))
        return self._real.rollback()

    def close(self):
        self._log.append(("close", self._index))
        return self._real.close()

    def __getattr__(self, name):
        return getattr(self._real, name)


class _FailingCursor:
    """Every statement goes to the real cursor, except the one whose text
    starts with ``fail_statement``: PostgreSQL itself is asked to raise in its
    place, so the error, the aborted transaction and the rollback are real."""

    def __init__(self, real, index, log, fail_statement):
        self._real = real
        self._index = index
        self._log = log
        self._fail_statement = fail_statement

    def execute(self, sql, params=None):
        normalized = " ".join(str(sql).split())
        if normalized.startswith(self._fail_statement):
            self._log.append(("injected", self._index))
            return self._real.execute(
                "DO $$ BEGIN RAISE EXCEPTION USING MESSAGE = %s, ERRCODE = 'P0001'; END $$",
                (_INJECTED_FAILURE,),
            )
        self._log.append(("execute", self._index, normalized))
        return self._real.execute(sql, params)

    def close(self):
        self._log.append(("cursor_close", self._index))
        return self._real.close()

    def __getattr__(self, name):
        return getattr(self._real, name)


class _Target:
    """One test's schema on the disposable database, a connector for the
    application code, and an independent observer connection."""

    def __init__(self, schema):
        self.schema = schema
        self.log = []
        self.handed_out = []
        self.fail_statement = None

    def raw_connection(self):
        import psycopg2

        return psycopg2.connect(_DSN)

    def connect_to_postgres(self):
        """Stands in for dbSetup.connect_to_postgres / main.connect_to_postgres.

        Same order as the real connector -- extension statement, commit, then
        register_vector -- plus the case schema's search_path, set and committed
        in that separate setup transaction so it survives. register_vector's
        catalog SELECT then opens the transaction init_db's DDL (or the route's
        INSERT) continues in, exactly as with the real connector."""
        from psycopg2 import sql
        from pgvector.psycopg2 import register_vector

        real = self.raw_connection()
        with real.cursor() as cursor:
            cursor.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(self.schema)))
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        real.commit()
        register_vector(real)
        index = len(self.handed_out)
        self.log.append(("connect", index))
        wrapped = _RecordingConnection(real, index, self.log, self.fail_statement)
        self.handed_out.append(wrapped)
        return wrapped

    def query(self, statement, params=None):
        """A fresh, independent connection: only committed data is visible."""
        connection = self.raw_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(statement, params)
                rows = cursor.fetchall() if cursor.description else None
            connection.commit()
            return rows
        finally:
            connection.close()

    def execute(self, statement, params=None):
        self.query(statement, params)

    def events(self, kind, index=None):
        return [entry for entry in self.log if entry[0] == kind and (index is None or entry[1] == index)]


@pytest.fixture()
def target():
    """A new, empty, uniquely named schema owned by the test role; dropped,
    with everything created in it, afterwards."""
    from psycopg2 import sql

    schema = f"app_test_{uuid.uuid4().hex[:10]}"
    case = _Target(schema)
    case.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        yield case
    finally:
        case.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))


def _configured_dimension():
    """The dimension the application starts with, from the pure default
    embedding-profile resolver and an explicit empty (sanitized) configuration.
    No embedder is called."""
    from rag_model_profiles import resolve_embedding_profile

    return resolve_embedding_profile({}).dimension


def _real_init_db(monkeypatch, case, *, fail_statement=None):
    import dbSetup

    case.fail_statement = fail_statement
    monkeypatch.setattr(dbSetup, "connect_to_postgres", case.connect_to_postgres)
    try:
        dbSetup.init_db(expected_dimension=_configured_dimension())
    finally:
        case.fail_statement = None


def _tables(case, schema=None):
    rows = case.query(
        "SELECT tablename FROM pg_tables WHERE schemaname = %s ORDER BY tablename", (schema or case.schema,)
    )
    return {row[0] for row in rows}


def _relations(case):
    rows = case.query(
        "SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = %s",
        (case.schema,),
    )
    return {row[0] for row in rows}


def _seed_user(case):
    from psycopg2 import sql

    rows = case.query(
        sql.SQL("INSERT INTO {}.users (username, password, token) VALUES (%s, %s, %s) RETURNING id::text").format(
            sql.Identifier(case.schema)
        ),
        (f"synthetic-{uuid.uuid4().hex[:8]}", "synthetic-not-a-password-hash", _SYNTHETIC_TOKEN),
    )
    return rows[0][0]


def _history_rows(case):
    from psycopg2 import sql

    return case.query(
        sql.SQL(
            "SELECT id, user_id::text, selected_text, question, answer, trace_id::text, answer_provenance "
            "FROM {}.chat_history ORDER BY id"
        ).format(sql.Identifier(case.schema))
    )


def _gate_marker_and_vector(case):
    marker = case.query("SELECT count(*) FROM public.synerge_gate_marker")[0][0]
    vector = case.query("SELECT extnamespace::regnamespace::text FROM pg_extension WHERE extname = 'vector'")
    return marker, vector


# --- the current /ask route, compiled from main.py ------------------------------

_ROUTE_DEFINITIONS = (
    "_CITATION_LIMITS",
    "_CLAIM_VERIFIER_PROMPT",
    "_CLAIM_VERIFIER_WORDS",
    "_claim_verifier_status",
    "_ollama_claim_verifier",
    "_MODEL_SELECTION_REFUSALS",
    "_resolve_answer_generation",
    "_resolve_citations",
    "ask_question",
)


def _defined_name(node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return node.name
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    return None


class _StreamedReply:
    """A streamed /api/generate response: NDJSON bytes in small chunks."""

    def __init__(self, records, chunk_size=11):
        data = "".join(json.dumps(record) + "\n" for record in records).encode("utf-8")
        self._chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, decode_unicode=True, chunk_size=32):
        yield from self._chunks


class _JsonReply:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        pass

    def json(self):
        return dict(self._body)


# A bounded, completed synthetic generation: done exactly true, done_reason
# exactly "stop", nonblank response text.
_FINAL_RECORD = {
    "response": "", "done": True, "done_reason": "stop",
    "prompt_eval_count": 321, "eval_count": 9,
    "total_duration": 2_000_000, "load_duration": 100_000,
    "prompt_eval_duration": 900_000, "eval_duration": 1_000_000,
}


class _Route:
    def __init__(self, client, calls, provenance):
        self.client = client
        self.calls = calls
        self.provenance = provenance

    def ask(self):
        response = self.client.post(
            "/ask",
            json={"question": _QUESTION, "auth_token": _SYNTHETIC_TOKEN, "mode": "document_qa", "selected_text": ""},
        )
        events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
        return response, events


def _load_ask_route(case, user_id):
    """The current ask_question and its main.py helpers, compiled from source
    into a namespace of real pure modules plus the stubs named in the module
    docstring, mounted on an isolated FastAPI app (no lifespan)."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from fastapi.testclient import TestClient

    import answer_provenance
    from answer_evidence import EvidenceBundle, EvidenceItem, EvidenceMode, EvidencePlanningResult
    from ask_stream import (
        STREAM_MEDIA_TYPE, delta_event, done_event, entry_id_event, error_event, evidence_event,
        verification_event,
    )
    from citation_generation import (
        AnswerMode, CitationGenerationResult, CitationLimits, CitationRegistry, build_generation_prompt,
        build_mode_prompt, citations_without_claims, generate_citations, safe_error,
    )
    import generation_profiles
    from ollama_embedding_provider import EmbeddingProviderError
    from ollama_generation import FAILURE_OUTPUT_LIMIT, OllamaGenerateStream, completed_generate_response
    from rag_model_profiles import resolve_embedding_profile
    from release_metadata import release_metadata_from_env
    from schemas import AskRequest

    tree = ast.parse(_MAIN_PATH.read_text(encoding="utf-8"), filename=str(_MAIN_PATH))
    selected = [node for node in tree.body if _defined_name(node) in _ROUTE_DEFINITIONS]
    assert sorted(_defined_name(node) for node in selected) == sorted(_ROUTE_DEFINITIONS), (
        "every route definition must exist exactly once in the current main.py"
    )

    calls = {"generate": [], "verify": [], "auth": [], "forbidden": []}
    provenance = []
    bundle = EvidenceBundle(
        mode=EvidenceMode.COMPLETE_DOCUMENT,
        items=(EvidenceItem(text=_EVIDENCE_TEXT, source_type="complete_document", document_id=1,
                            filename="synthetic-agreement.txt"),),
    )

    def resolve_caller_id(auth_token):
        calls["auth"].append(auth_token)
        assert auth_token == _SYNTHETIC_TOKEN
        return user_id

    def post_ollama(endpoint, payload, *, stream=False, timeout=60):
        assert endpoint == "/api/generate"
        if stream:
            calls["generate"].append({"payload": payload, "timeout": timeout})
            return _StreamedReply([*({"response": token} for token in _ANSWER_TOKENS), _FINAL_RECORD])
        calls["verify"].append({"payload": payload, "timeout": timeout})
        return _JsonReply({"response": "supported", "done": True, "done_reason": "stop"})

    def forbidden(name):
        def refuse(*args, **kwargs):
            calls["forbidden"].append(name)
            raise AssertionError(f"{name} must not be called for a default answer")

        return refuse

    def recording_build_answer_provenance(**kwargs):
        record = answer_provenance.build_answer_provenance(**kwargs)
        provenance.append(record)
        return record

    def durable_entry_id_event(entry_id):
        # Emitted only after the commit: the row is already visible to an
        # independent connection when its id is announced.
        from psycopg2 import sql

        rows = case.query(
            sql.SQL("SELECT count(*) FROM {}.chat_history WHERE id = %s").format(sql.Identifier(case.schema)),
            (entry_id,),
        )
        case.log.append(("entry_id_event", entry_id, rows[0][0]))
        return entry_id_event(entry_id)

    app = FastAPI(title="isolated /ask integration route")
    namespace = {
        "app": app,
        "AskRequest": AskRequest,
        "Optional": Optional,
        "re": re,
        "datetime": datetime,
        "HTTPException": HTTPException,
        "StreamingResponse": StreamingResponse,
        "STREAM_MEDIA_TYPE": STREAM_MEDIA_TYPE,
        "EmbeddingProviderError": EmbeddingProviderError,
        # Real pure modules.
        "AnswerGenerationSnapshot": generation_profiles.AnswerGenerationSnapshot,
        "ModelSelectionError": generation_profiles.ModelSelectionError,
        "OverrideNotPermittedError": generation_profiles.OverrideNotPermittedError,
        "UnknownModelProfileError": generation_profiles.UnknownModelProfileError,
        "ProfileNotQualifiedError": generation_profiles.ProfileNotQualifiedError,
        "ModelIdentityMismatchError": generation_profiles.ModelIdentityMismatchError,
        "SelectionAuthorityUnavailableError": generation_profiles.SelectionAuthorityUnavailableError,
        "ModelIdentityUnavailableError": generation_profiles.ModelIdentityUnavailableError,
        "resolve_answer_generation": generation_profiles.resolve_answer_generation,
        "_MODEL_REGISTRY": generation_profiles.default_model_registry(),
        "_RELEASE": release_metadata_from_env({}),
        "_EMBEDDING_PROFILE": resolve_embedding_profile({}),
        "AnswerMode": AnswerMode,
        "CitationGenerationResult": CitationGenerationResult,
        "CitationLimits": CitationLimits,
        "CitationRegistry": CitationRegistry,
        "build_generation_prompt": build_generation_prompt,
        "build_mode_prompt": build_mode_prompt,
        "citations_without_claims": citations_without_claims,
        "generate_citations": generate_citations,
        "safe_error": safe_error,
        "OllamaGenerateStream": OllamaGenerateStream,
        "FAILURE_OUTPUT_LIMIT": FAILURE_OUTPUT_LIMIT,
        "completed_generate_response": completed_generate_response,
        "AnswerTimer": answer_provenance.AnswerTimer,
        "new_trace_id": answer_provenance.new_trace_id,
        "build_answer_provenance": recording_build_answer_provenance,
        "answer_provenance_json": answer_provenance.answer_provenance_json,
        "evidence_event": evidence_event,
        "delta_event": delta_event,
        "verification_event": verification_event,
        "entry_id_event": durable_entry_id_event,
        "error_event": error_event,
        "done_event": done_event,
        # The disposable target.
        "connect_to_postgres": case.connect_to_postgres,
        # Mocked boundaries.
        "_resolve_caller_id": resolve_caller_id,
        "_caller_is_admin": forbidden("_caller_is_admin"),
        "_installed_model_digest": forbidden("_installed_model_digest"),
        "_resolve_authorized_scope": lambda auth_token: "synthetic-authorized-scope",
        "_build_answer_evidence_planner": lambda scope: type(
            "_Planner", (), {"plan": staticmethod(lambda request, scope: EvidencePlanningResult(bundle=bundle))}
        )(),
        "_evidence_request_from_ask": lambda request, scope: request,
        "post_ollama": post_ollama,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(_MAIN_PATH), "exec"), namespace)
    return _Route(TestClient(app), calls, provenance)


def _route_connection_indexes(case, before):
    return [entry[1] for entry in case.events("connect") if entry[1] >= before]


# --- A. real startup on a fresh schema, then again over existing rows -----------


@needs_db
def test_real_init_db_creates_expected_schema_and_is_repeatable(monkeypatch, target):
    from psycopg2 import sql

    dimension = _configured_dimension()
    print(f"configured embedding dimension from resolve_embedding_profile({{}}): {dimension}")
    assert dimension == 1024, "the configured default profile, not the legacy 768 module default"

    _real_init_db(monkeypatch, target)

    # One setup commit in the connector (search_path), then init_db's own
    # DDL transaction: exactly one commit, no rollback, connection closed.
    [first] = target.handed_out
    assert target.events("commit", 0) == [("commit", 0)]
    assert target.events("rollback", 0) == []
    assert target.events("close", 0) == [("close", 0)]
    assert first.real.closed

    assert _tables(target) == _APPLICATION_TABLES
    assert _tables(target, "public") == {"synerge_gate_marker"}, "no application table in public"
    pgcrypto = target.query(
        "SELECT extnamespace::regnamespace::text, pg_get_userbyid(extowner) = current_user "
        "FROM pg_extension WHERE extname = 'pgcrypto'"
    )
    assert pgcrypto == [(target.schema, True)], "init_db created pgcrypto (here: as the test role, first on the path)"
    dimensions = target.query(
        "SELECT c.relname, a.atttypmod, format_type(a.atttypid, a.atttypmod) FROM pg_attribute a "
        "JOIN pg_class c ON c.oid = a.attrelid JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = %s AND a.attname = 'embedding' AND NOT a.attisdropped ORDER BY c.relname",
        (target.schema,),
    )
    assert dimensions == [
        ("document_chunks", dimension, f"vector({dimension})"),
        ("knowledge_base", dimension, f"vector({dimension})"),
    ]
    provenance_columns = target.query(
        "SELECT column_name, data_type, is_nullable, column_default FROM information_schema.columns "
        "WHERE table_schema = %s AND table_name = 'chat_history' AND column_name IN ('trace_id', 'answer_provenance') "
        "ORDER BY column_name",
        (target.schema,),
    )
    assert provenance_columns == [("answer_provenance", "jsonb", "YES", None), ("trace_id", "uuid", "YES", None)]

    # An existing account and history row, then startup again.
    user_id = _seed_user(target)
    target.execute(
        sql.SQL(
            "INSERT INTO {}.chat_history (ts, selected_text, question, answer, user_id) VALUES (%s, %s, %s, %s, %s)"
        ).format(sql.Identifier(target.schema)),
        ("2026-09-01T10:00:00", "", "A synthetic earlier question?", "A synthetic earlier answer.", user_id),
    )
    snapshot = sql.SQL(
        "SELECT u.id::text, u.username, u.token, u.xmin::text, h.id, h.question, h.answer, h.trace_id, "
        "h.answer_provenance, h.xmin::text FROM {schema}.users u JOIN {schema}.chat_history h ON h.user_id = u.id"
    ).format(schema=sql.Identifier(target.schema))
    before = target.query(snapshot)
    assert len(before) == 1

    _real_init_db(monkeypatch, target)

    second = target.handed_out[1]
    assert target.events("commit", 1) == [("commit", 1)] and target.events("rollback", 1) == []
    assert target.events("close", 1) == [("close", 1)] and second.real.closed
    after = target.query(snapshot)
    assert after == before, "the row's values and tuple version (xmin) are unchanged: nothing was rewritten"
    assert after[0][7:9] == (None, None), "a legacy answer stays 'not recorded'; nothing is backfilled"
    assert _tables(target) == _APPLICATION_TABLES


# --- B. a real database error late in startup rolls the whole attempt back -------


@needs_db
def test_real_init_db_rolls_back_schema_ddl_on_database_failure(monkeypatch, target):
    import psycopg2

    marker_before, vector_before = _gate_marker_and_vector(target)

    with pytest.raises(psycopg2.Error) as excinfo:
        _real_init_db(monkeypatch, target, fail_statement="CREATE TABLE IF NOT EXISTS document_insights")

    assert excinfo.value.pgcode == "P0001" and _INJECTED_FAILURE in str(excinfo.value), (
        "the failure is PostgreSQL's own error, raised by init_db's real error handling"
    )
    executed = [entry[2] for entry in target.log if entry[0] == "execute"]
    for statement in ('CREATE EXTENSION IF NOT EXISTS "pgcrypto";', "CREATE TABLE IF NOT EXISTS users",
                      "CREATE TABLE IF NOT EXISTS chat_history", "CREATE TABLE IF NOT EXISTS knowledge_base",
                      "CREATE TABLE IF NOT EXISTS admin_audit_log"):
        assert any(sql_text.startswith(statement) for sql_text in executed), f"ran before the failure: {statement}"
    assert target.events("injected", 0) == [("injected", 0)], "the failure was late: after every earlier DDL"

    [connection] = target.handed_out
    assert target.events("commit", 0) == [], "init_db never committed"
    order = [entry[0] for entry in target.log if entry[0] in ("injected", "rollback", "cursor_close", "close")]
    assert order == ["injected", "rollback", "cursor_close", "close"]
    assert connection.real.closed

    # An independent connection sees nothing from the attempt.
    assert _tables(target) == set()
    assert _relations(target) == set(), "no table, sequence, index or type from the attempt survived"
    assert target.query("SELECT count(*) FROM pg_extension WHERE extname = 'pgcrypto'")[0][0] == 0
    assert _gate_marker_and_vector(target) == (marker_before, vector_before) == (1, [("public",)])


# --- C. the current /ask route saves a completed answer with its provenance -------


@needs_db
def test_real_ask_persists_real_provenance_with_answer_and_entry_id(monkeypatch, target):
    from generation_profiles import LLAMA_CONTROL_DIGEST
    from rag_model_profiles import resolve_embedding_profile
    from release_metadata import release_metadata_from_env

    _real_init_db(monkeypatch, target)
    user_id = _seed_user(target)
    route = _load_ask_route(target, user_id)
    first_route_connection = len(target.handed_out)

    response, events = route.ask()

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert events[-1] == {"type": "done", "ok": True}
    assert not [event for event in events if event["type"] == "error"]
    [entry_id] = [event["entry_id"] for event in events if event["type"] == "entry_id"]
    [evidence] = [event for event in events if event["type"] == "evidence"]
    assert evidence["generation"] == {"profile_id": "llama_control", "model": "llama3.1:8b"}
    assert "".join(event["text"] for event in events if event["type"] == "delta") == "".join(_ANSWER_TOKENS)

    # The durable row, read back independently, is this answer.
    [row] = _history_rows(target)
    row_id, row_user, selected_text, question, answer, trace_id, stored = row
    assert row_id == entry_id
    assert (row_user, selected_text, question, answer) == (user_id, "", _QUESTION, "".join(_ANSWER_TOKENS))

    # ...and its trace id and JSONB are the provenance the route really built.
    [built] = route.provenance
    assert trace_id == built["trace_id"] and str(uuid.UUID(trace_id)) == trace_id
    assert uuid.UUID(trace_id).version == 4
    assert stored == json.loads(json.dumps(built)), "JSONB holds exactly the constructed record"
    generation = stored["generation"]
    assert (generation["profile_id"], generation["model_tag"], generation["selection"]) == (
        "llama_control", "llama3.1:8b", "default",
    )
    assert generation["expected_digest"] == LLAMA_CONTROL_DIGEST
    assert generation["observed_digest"] is None, "no inventory lookup is made for the pinned control"
    assert generation["requested_options"] == {"temperature": 0.2, "num_ctx": 8192, "num_predict": 1200}
    assert generation["think"] is None
    release = release_metadata_from_env({})
    assert stored["release"] == {"app_version": release.app_version, "build_sha": None}
    profile = resolve_embedding_profile({})
    assert stored["retrieval"] == {
        "evidence_mode": "complete_document",
        "configured_embedding_profile": {
            "profile_id": profile.profile_id, "provider": profile.provider,
            "model": profile.model, "dimension": profile.dimension,
        },
    }
    assert stored["verification"]["calls_attempted"] == len(route.calls["verify"]) == 1
    assert stored["verification"]["model_tag"] == "llama3.1:8b"
    # Schema version 2: the verifier settings that actually ran, as stored in
    # real JSONB and read back -- the control's pin, no observation, no think
    # field, and the 30 s verifier read timeout that reached the transport.
    assert stored["schema_version"] == 2
    added = ("expected_digest", "observed_digest", "think", "read_timeout_s")
    assert {key: stored["verification"][key] for key in added} == {
        "expected_digest": LLAMA_CONTROL_DIGEST, "observed_digest": None, "think": None, "read_timeout_s": 30.0,
    }
    [verify_call] = route.calls["verify"]
    assert verify_call["timeout"] == stored["verification"]["read_timeout_s"]
    assert "think" not in verify_call["payload"]
    assert verify_call["payload"]["options"] == stored["verification"]["requested_options"]
    assert stored["ollama_metrics"]["done_reason"] == "stop"
    assert (stored["ollama_metrics"]["prompt_eval_count"], stored["ollama_metrics"]["eval_count"]) == (321, 9)

    # The id is announced only after the commit, and the connection is closed.
    route_connections = _route_connection_indexes(target, first_route_connection)
    assert len(route_connections) == 1
    [index] = route_connections
    commit_at = target.log.index(("commit", index))
    close_at = target.log.index(("close", index))
    announced = [i for i, entry in enumerate(target.log) if entry[0] == "entry_id_event"]
    assert len(announced) == 1 and commit_at < close_at < announced[0]
    assert target.log[announced[0]] == ("entry_id_event", entry_id, 1), "durable when announced"
    assert target.events("rollback", index) == []
    assert target.handed_out[index].real.closed
    assert route.calls["auth"] == [_SYNTHETIC_TOKEN] and route.calls["forbidden"] == []
    assert len(route.calls["generate"]) == 1
    assert route.calls["generate"][0]["payload"]["model"] == "llama3.1:8b"


# --- D. a real INSERT failure saves nothing and announces nothing -----------------


@needs_db
def test_real_ask_database_failure_emits_no_entry_id_and_saves_no_row(monkeypatch, target):
    from psycopg2 import sql

    _real_init_db(monkeypatch, target)
    user_id = _seed_user(target)
    schema = sql.Identifier(target.schema)
    function = sql.SQL("{}.{}").format(schema, sql.Identifier("gate_reject_chat_history_insert"))
    target.execute(
        sql.SQL(
            "CREATE FUNCTION {} () RETURNS trigger LANGUAGE plpgsql AS "
            "$$ BEGIN RAISE EXCEPTION '" + _INSERT_FAILURE + "'; END $$"
        ).format(function)
    )
    target.execute(
        sql.SQL("CREATE TRIGGER gate_reject_insert BEFORE INSERT ON {}.chat_history FOR EACH ROW EXECUTE FUNCTION {} ()")
        .format(schema, function)
    )
    try:
        route = _load_ask_route(target, user_id)
        first_route_connection = len(target.handed_out)

        response, events = route.ask()

        assert response.status_code == 200
        assert events[-1] == {"type": "done", "ok": False}
        assert not [event for event in events if event["type"] == "entry_id"]
        errors = [event for event in events if event["type"] == "error"]
        assert [error["code"] for error in errors] == ["internal_error"]
        assert _INSERT_FAILURE not in response.text and "chat_history" not in response.text, "a safe error only"
        assert [event for event in events if event["type"] == "verification"], "verification ran before persistence"

        assert len(route.provenance) == 1, "the provenance was built; the INSERT failed in PostgreSQL"
        assert _history_rows(target) == [], "no durable answer row from the attempt"
        [index] = _route_connection_indexes(target, first_route_connection)
        assert target.events("commit", index) == []
        order = [entry[0] for entry in target.log if entry[0] in ("rollback", "close") and entry[1] == index]
        assert order == ["rollback", "close"]
        assert target.handed_out[index].real.closed
        assert not [entry for entry in target.log if entry[0] == "entry_id_event"]
    finally:
        target.execute(sql.SQL("DROP TRIGGER IF EXISTS gate_reject_insert ON {}.chat_history").format(schema))
        target.execute(sql.SQL("DROP FUNCTION IF EXISTS {} ()").format(function))
