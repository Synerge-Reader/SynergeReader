"""Unit contracts for generation_selection.py: the durable selection store's
logic and the single-owner manager.

No database, model, network or subprocess: the PostgreSQL session is a
scripted in-memory connection that records every statement, and the owner
manager runs on real threads against fake stores. The SQL itself -- advisory
lock exclusivity, compare-and-set, recovery and fencing on a real server -- is
proven in tests/test_generation_selection_integration.py (needs
SYNERGE_MIGRATION_TEST_DSN).

What these prove: constructing a store opens nothing; a store whose advisory
lock is held elsewhere closes its session and reports "not owner"; the schema
check refuses same-named objects of another shape and accepts its own; the
selection row's invariants are enforced on read; EVERY write after acquisition
carries the owner fence (owner token and sequence AND this backend's advisory
lock); a dispatch intent that is refused or ambiguous fails closed with a
fixed message; the owner thread retries while another process owns, installs
exactly what it recovered (unresolved calls keep their capacity; an
interrupted switch keeps admission paused), never takes ownership over an
incompatible schema, fences the ledger for good when ownership is lost or the
application stops, and never re-acquires in the same process.
"""

import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from generation_admission import (  # noqa: E402
    AdmissionRefusal,
    AdmissionRefused,
    AdmissionSettings,
    CallState,
    DispatchIntent,
    DispatchIntents,
    DispatchRefused,
    GenerationLedger,
    JobKind,
    SharedSelection,
)
from generation_selection import (  # noqa: E402
    EXPECTED_SCHEMA,
    OWNER_LOCK_KEY,
    DispatchIntentRefused,
    GenerationOwner,
    GenerationStoreError,
    OwnerRecovery,
    OwnerSettings,
    OwnerState,
    PostgresGenerationStore,
    SchemaIncompatible,
    SelectionInvalid,
    SelectionRecord,
    validate_generation_schema,
)

BOOT = "00000000-0000-4000-8000-000000000001"
SWITCH = "00000000-0000-4000-8000-0000000000aa"
OTHER_BOOT = "00000000-0000-4000-8000-0000000000bb"
_WAIT = 5.0


def _norm(sql):
    return " ".join(str(sql).split())


class _Cursor:
    def __init__(self, connection):
        self._connection = connection
        self._rows = None
        self.description = None

    def execute(self, sql, params=None):
        self._connection.statements.append((_norm(sql), params))
        rows = self._connection.respond(_norm(sql), params)
        self._rows = rows
        self.description = None if rows is None else ("column",)

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows or [])

    def close(self):
        pass


class _Connection:
    """A scripted owner session. ``locked_elsewhere`` makes the advisory lock
    unavailable; ``columns`` overrides one table's catalog shape; ``write_rows``
    is what fenced writes return (None: the fence did not match)."""

    def __init__(self, *, locked_elsewhere=False, columns=None, write_rows="ok", raise_on=None):
        self.statements = []
        self.locked_elsewhere = locked_elsewhere
        self.columns = columns or {}
        self.write_rows = write_rows
        self.raise_on = raise_on
        self.autocommit = False
        self.commits = self.rollbacks = self.closes = 0
        self.selection_row = ("llama_control", None, "active", 1, None, None, None, None)

    def cursor(self):
        return _Cursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closes += 1

    def respond(self, sql, params):
        if self.raise_on and sql.startswith(self.raise_on):
            raise RuntimeError("SECRET-DETAIL server closed the connection unexpectedly")
        if sql.startswith("SELECT pg_try_advisory_lock"):
            return [(not self.locked_elsewhere,)]
        if sql.startswith("SELECT c.relkind"):
            return [("r",)]
        if sql.startswith("SELECT a.attname, format_type"):
            columns = self.columns.get(params[0], EXPECTED_SCHEMA[params[0]][0])
            return [(name, kind, nullable) for name, (kind, nullable) in columns.items()]
        if sql.startswith("SELECT a.attname FROM pg_index"):
            return [(EXPECTED_SCHEMA[params[0]][1],)]
        if sql.startswith("INSERT INTO generation_owner"):
            return [(7,)]
        if sql.startswith("SELECT active_profile_id"):
            return [self.selection_row]
        if sql.startswith("SELECT job_kind"):
            return [("answer", 2, 3), ("helper", 1, 1)]
        if sql.startswith("INSERT INTO generation_calls"):
            return None if self.write_rows is None else [(params[0],)]
        if sql.startswith(("UPDATE generation_selection", "UPDATE generation_owner")):
            if self.write_rows is None:
                return []
            if sql.startswith("UPDATE generation_owner"):
                return [(7,)]
            return [self.selection_row]
        return None


def _store(connection):
    return PostgresGenerationStore(lambda: connection, boot_id=BOOT)


def _intent():
    return DispatchIntent(call_id="00000000-0000-4000-8000-0000000000c1",
                          job_id="00000000-0000-4000-8000-0000000000d1", job_kind="answer", call_role="answer",
                          profile_id="llama_control", model_tag="llama3.1:8b", selection_epoch=1)


# --- the store -----------------------------------------------------------------------


def test_constructing_a_store_opens_nothing_and_validates_its_arguments():
    opened = []
    PostgresGenerationStore(lambda: opened.append(1), boot_id=BOOT)
    assert opened == []
    for kwargs in (dict(boot_id="not-a-uuid"), dict(boot_id=BOOT, lock_key=(1, 2**31)),
                   dict(boot_id=BOOT, statement_timeout_ms=0)):
        with pytest.raises(ValueError):
            PostgresGenerationStore(lambda: None, **kwargs)


def test_a_lock_held_elsewhere_means_not_owner_and_the_session_is_closed():
    connection = _Connection(locked_elsewhere=True)
    store = _store(connection)
    assert store.try_acquire("llama_control") is None
    assert connection.closes == 1 and connection.autocommit is True
    assert [sql for sql, _ in connection.statements] == [
        "SET statement_timeout = %s", "SELECT pg_try_advisory_lock(%s, %s)",
    ], "no DDL, seed or owner change without the lock"
    assert connection.statements[1][1] == OWNER_LOCK_KEY
    with pytest.raises(GenerationStoreError):
        store.record_dispatch(_intent())


def test_acquisition_creates_validates_seeds_and_recovers_in_one_transaction():
    connection = _Connection()
    store = _store(connection)
    recovery = store.try_acquire("llama_control")
    assert recovery == OwnerRecovery(
        owner_seq=7, selection=SelectionRecord("llama_control", None, "active", 1, None, None, None, None),
        unresolved_jobs={JobKind.ANSWER: 2, JobKind.HELPER: 1}, unresolved_calls=4,
    )
    statements = [sql for sql, _ in connection.statements]
    assert sum(sql.startswith("CREATE TABLE IF NOT EXISTS generation_") for sql in statements) == 3
    seed = next(sql for sql in statements if sql.startswith("INSERT INTO generation_selection"))
    assert "ON CONFLICT (singleton) DO NOTHING" in seed, "an existing selection is never overwritten"
    assert connection.commits == 1 and connection.autocommit is True
    assert not [sql for sql in statements if sql.startswith(("DROP", "ALTER", "TRUNCATE", "DELETE"))]
    assert not [sql for sql in statements if "generation_calls" in sql and sql.startswith("UPDATE")], (
        "startup never clears or resolves unresolved calls"
    )


@pytest.mark.parametrize(
    "table, change",
    [("generation_calls", "missing-column"), ("generation_selection", "wrong-type"),
     ("generation_owner", "extra-column")],
)
def test_a_same_named_object_of_another_shape_is_refused(table, change):
    columns = dict(EXPECTED_SCHEMA[table][0])
    if change == "missing-column":
        columns.pop("state")
    elif change == "wrong-type":
        columns["epoch"] = ("integer", False)
    else:
        columns["note"] = ("text", True)
    connection = _Connection(columns={table: columns})
    with pytest.raises(SchemaIncompatible):
        _store(connection).try_acquire("llama_control")
    assert connection.commits == 0 and connection.rollbacks >= 2 and connection.closes == 1


def test_the_schema_check_needs_ordinary_tables_with_the_expected_primary_keys():
    class _Catalog(_Connection):
        def respond(self, sql, params):
            if sql.startswith("SELECT c.relkind"):
                return [("v",)] if params[0] == "generation_owner" else [("r",)]
            return super().respond(sql, params)

    cursor = _Catalog().cursor()
    with pytest.raises(SchemaIncompatible):
        validate_generation_schema(cursor)
    validate_generation_schema(_Connection().cursor())


@pytest.mark.parametrize(
    "row",
    [("Llama!", None, "active", 1, None, None, None, None),
     ("llama_control", None, "active", 0, None, None, None, None),
     ("llama_control", "gemma_candidate", "active", 1, None, None, None, None),
     ("llama_control", None, "switching", 2, None, None, None, None),
     ("llama_control", None, "paused", 1, None, None, None, None),
     ("llama_control", None, "active", 1, None, None, None, "aborted:because")],
    ids=["bad-profile", "epoch-zero", "active-with-target", "switching-without-switch", "unknown-state",
         "unknown-outcome"],
)
def test_the_selection_row_invariants_are_enforced(row):
    with pytest.raises(SelectionInvalid):
        SelectionRecord(*row)


def test_every_write_after_acquisition_carries_the_owner_fence():
    connection = _Connection()
    store = _store(connection)
    store.try_acquire("llama_control")
    connection.statements.clear()
    switching = SelectionRecord("llama_control", "gemma_candidate", "switching", 2, SWITCH, BOOT, None, None)
    connection.selection_row = ("llama_control", "gemma_candidate", "switching", 2, SWITCH, BOOT, None, None)
    store.record_dispatch(_intent())
    store.record_outcome(_intent(), CallState.OBSERVED_TERMINAL)
    store.record_outcome(_intent(), CallState.UNKNOWN)
    store.begin_switch(expected=SelectionRecord("llama_control", None, "active", 1, None, None, None, None),
                       target_profile_id="gemma_candidate", switch_id=SWITCH)
    store.commit_switch(switching=switching)
    store.abort_switch(switching=switching, reason="admin_abort")
    assert store.check_owner()
    writes = [(sql, params) for sql, params in connection.statements
              if sql.startswith(("INSERT", "UPDATE", "DELETE"))]
    assert len(writes) == 7
    for sql, params in writes:
        assert "o.owner_token = %s::uuid AND o.owner_seq = %s" in sql, sql
        assert "pg_locks" in sql and "l.pid = pg_backend_pid()" in sql and "l.granted" in sql, sql
        assert params[-2:] == OWNER_LOCK_KEY and params[-3] == 7
    commit_sql = writes[4][0]
    for column in ("state = 'switching'", "target_profile_id = %s", "epoch = %s", "switch_id = %s::uuid",
                   "switch_owner_boot_id = %s::uuid"):
        assert column in commit_sql, f"the commit compare-and-set matches {column}"


def test_a_refused_or_ambiguous_dispatch_intent_fails_closed_with_a_fixed_message():
    connection = _Connection(write_rows=None)
    store = _store(connection)
    store.try_acquire("llama_control")
    with pytest.raises(DispatchIntentRefused) as refused:
        store.record_dispatch(_intent())
    assert "owner" in str(refused.value)

    ambiguous = _Connection(raise_on="INSERT INTO generation_calls")
    store = _store(ambiguous)
    store.try_acquire("llama_control")
    with pytest.raises(DispatchIntentRefused) as refused:
        store.record_dispatch(_intent())
    assert "SECRET" not in str(refused.value) and refused.value.__cause__ is None
    assert refused.value.__suppress_context__
    assert store.check_owner() is True, "the session itself is still alive"


def test_release_gives_up_the_lock_and_later_writes_are_refused():
    connection = _Connection()
    store = _store(connection)
    store.try_acquire("llama_control")
    store.release()
    store.release()
    assert connection.closes == 1
    assert ("SELECT pg_advisory_unlock(%s, %s)", OWNER_LOCK_KEY) in connection.statements
    assert store.check_owner() is False
    with pytest.raises(GenerationStoreError):
        store.record_dispatch(_intent())


# --- the owner manager -----------------------------------------------------------------


class _FakeStore(DispatchIntents):
    def __init__(self, outcome, alive=True):
        self.outcome = outcome
        self.alive = alive
        self.released = 0

    def try_acquire(self, default_profile_id):
        self.default_profile_id = default_profile_id
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome

    def check_owner(self):
        return self.alive() if callable(self.alive) else self.alive

    def release(self):
        self.released += 1

    def record_dispatch(self, intent):
        pass

    def record_outcome(self, intent, state):
        pass


def _recovery(record=None, answers=0, helpers=0):
    record = record or SelectionRecord("llama_control", None, "active", 4, None, None, None, None)
    return OwnerRecovery(owner_seq=3, selection=record, unresolved_jobs={JobKind.ANSWER: answers,
                                                                          JobKind.HELPER: helpers},
                         unresolved_calls=answers + helpers)


def _owner(stores, **kwargs):
    ledger = GenerationLedger(AdmissionSettings(), clock=time.monotonic)
    made = []
    logs = []

    def factory():
        store = stores.pop(0) if stores else _FakeStore(None)
        made.append(store)
        return store

    owner = GenerationOwner(store_factory=factory, ledger=ledger, default_profile_id=lambda: "llama_control",
                            settings=OwnerSettings(retry_s=0.01, heartbeat_s=0.01), log=logs.append, **kwargs)
    return owner, ledger, made, logs


def _until(predicate):
    deadline = time.monotonic() + _WAIT
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        time.sleep(0.005)


def test_the_owner_retries_while_another_process_owns_then_installs_what_it_recovered():
    stores = [_FakeStore(None), _FakeStore(GenerationStoreError("SECRET")), _FakeStore(_recovery(answers=1, helpers=1))]
    owner, ledger, made, logs = _owner(stores)
    with pytest.raises(AdmissionRefused) as refused:
        ledger.enqueue_answer(lambda: None)
    assert refused.value.reason is AdmissionRefusal.UNAVAILABLE, "nothing is admitted before ownership"
    assert owner.start() and not owner.start()
    _until(lambda: owner.state is OwnerState.OWNER)
    snap = ledger.snapshot()
    assert snap.selection == SharedSelection("llama_control", 4) and snap.admission == "open"
    assert (snap.unresolved_answers, snap.unresolved_helpers) == (1, 1), "recovered work keeps its capacity"
    assert [store.released for store in made[:2]] == [1, 1] and owner.store is made[2]
    assert all("SECRET" not in line for line in logs)
    owner.stop()
    _until(lambda: owner.state is OwnerState.STOPPED)
    assert made[2].released == 1 and ledger.fenced


def test_an_interrupted_switch_is_recovered_paused_never_reset():
    interrupted = SelectionRecord("llama_control", "gemma_candidate", "switching", 5, SWITCH, OTHER_BOOT, None, None)
    owner, ledger, _, _ = _owner([_FakeStore(_recovery(interrupted))])
    owner.start()
    _until(lambda: owner.state is OwnerState.OWNER)
    snap = ledger.snapshot()
    assert (snap.admission, snap.paused_by, snap.selection) == ("paused", SWITCH, SharedSelection("llama_control", 5))
    owner.stop()


@pytest.mark.parametrize("error", [SchemaIncompatible("x"), SelectionInvalid("x")])
def test_an_incompatible_schema_or_selection_never_becomes_the_owner(error):
    owner, ledger, made, _ = _owner([_FakeStore(error) for _ in range(50)])
    owner.start()
    _until(lambda: len(made) >= 3)
    assert owner.state is OwnerState.SCHEMA_INCOMPATIBLE and owner.store is None
    assert ledger.snapshot().admission == "unavailable"
    owner.stop()
    _until(lambda: owner.state is OwnerState.STOPPED)
    assert all(store.released == 1 for store in made)


def test_lost_ownership_fences_for_good_and_is_never_re_acquired():
    alive = {"value": True}
    stores = [_FakeStore(_recovery(), alive=lambda: alive["value"]), _FakeStore(_recovery())]
    owner, ledger, made, logs = _owner(stores)
    owner.start()
    _until(lambda: owner.state is OwnerState.OWNER)
    running = ledger.try_acquire_helper()
    assert running is not None
    alive["value"] = False
    _until(lambda: owner.state is OwnerState.LOST)
    assert ledger.fenced and owner.store is None and made[0].released == 1
    time.sleep(0.05)
    assert len(made) == 1, "a lost owner never acquires again in this process"
    running.bind_snapshot(type("S", (), {"profile_id": "llama_control", "selection_epoch": 4,
                                         "model_tag": "llama3.1:8b"})())
    with pytest.raises(DispatchRefused):
        running.dispatching("helper")
    assert ledger.release(running) is CallState.NEVER_DISPATCHED
    assert any("lost" in line for line in logs)


def test_a_stopped_owner_is_never_started_and_a_failed_thread_start_fences():
    owner, ledger, made, _ = _owner([])
    owner.stop()
    assert owner.start() is False and made == [] and ledger.fenced

    def no_threads(*args, **kwargs):
        raise RuntimeError("SECRET-DETAIL")

    owner, ledger, made, logs = _owner([], thread_factory=no_threads)
    assert owner.start() is False
    assert owner.state is OwnerState.STOPPED and ledger.fenced and made == []
    assert all("SECRET" not in line for line in logs)
