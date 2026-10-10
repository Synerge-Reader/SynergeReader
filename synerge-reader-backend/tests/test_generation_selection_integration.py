"""The durable selection store against a disposable PostgreSQL.

Opt-in integration layer, like tests/test_startup_and_answer_persistence_integration.py:
every test here skips unless SYNERGE_MIGRATION_TEST_DSN names a throw-away
PostgreSQL database. Nothing runs at collection time beyond importing pure
modules.

Each test works inside its own freshly created schema (``app_test_<hex>``),
first on the search_path, and uses its own advisory-lock key derived from that
schema, so no test can see another's tables or ownership. Every connection a
test opens is closed and the schema is dropped afterwards.

What is REAL here: generation_selection.PostgresGenerationStore -- its DDL,
schema validation, session advisory lock, owner-token fence, dispatch-intent
writes, recovery reads and every compare-and-set -- executed by real psycopg2
sessions against a real server. Two stores in one test are two independent
sessions, i.e. two processes as far as PostgreSQL is concerned.

What is NOT qualified here: the application's own connector and environment,
a transaction-pooling proxy (which would break session advisory locks), the
production PostgreSQL version, network partitions with half-open TCP sessions,
or anything about Ollama.
"""

import os
import sys
import uuid

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from generation_admission import CallState, DispatchIntent, JobKind  # noqa: E402
from generation_selection import (  # noqa: E402
    DispatchIntentRefused,
    PostgresGenerationStore,
    SchemaIncompatible,
    SelectionRecord,
)

_DSN = os.environ.get("SYNERGE_MIGRATION_TEST_DSN")
needs_db = pytest.mark.skipif(not _DSN, reason="set SYNERGE_MIGRATION_TEST_DSN to run the database tests")

_TABLES = {"generation_selection", "generation_owner", "generation_calls"}


class _Target:
    def __init__(self):
        self.schema = f"app_test_{uuid.uuid4().hex[:10]}"
        self.lock_key = (0x53594E52, int(self.schema[-7:], 16))
        self.opened = []

    def raw(self):
        import psycopg2

        connection = psycopg2.connect(_DSN)
        self.opened.append(connection)
        return connection

    def connect(self):
        """What the application's connector hands the store: a connection
        with a transaction left open, here on the test schema's search_path."""
        from psycopg2 import sql

        connection = self.raw()
        with connection.cursor() as cursor:
            cursor.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(self.schema)))
        connection.commit()
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")  # leaves a transaction open, as register_vector does
        return connection

    def store(self, boot_id=None):
        return PostgresGenerationStore(self.connect, boot_id=boot_id or str(uuid.uuid4()), lock_key=self.lock_key)

    def query(self, statement, params=None):
        from psycopg2 import sql

        connection = self.raw()
        try:
            with connection.cursor() as cursor:
                cursor.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(self.schema)))
                cursor.execute(statement, params)
                rows = cursor.fetchall() if cursor.description else None
            connection.commit()
            return rows
        finally:
            connection.close()

    def tables(self):
        rows = self.query("SELECT tablename FROM pg_tables WHERE schemaname = %s", (self.schema,))
        return {row[0] for row in rows}


@pytest.fixture()
def target():
    from psycopg2 import sql

    case = _Target()
    case.query(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(case.schema)))
    try:
        yield case
    finally:
        for connection in case.opened:
            try:
                connection.close()
            except Exception:
                pass
        case.query(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(case.schema)))


def _intent(kind="answer", role="answer", job_id=None):
    return DispatchIntent(call_id=str(uuid.uuid4()), job_id=job_id or str(uuid.uuid4()), job_kind=kind,
                          call_role=role, profile_id="llama_control", model_tag="llama3.1:8b", selection_epoch=1)


def _calls(case):
    return case.query("SELECT call_role, state, owner_seq FROM generation_calls ORDER BY created_at, call_role")


# --- schema ----------------------------------------------------------------------------


@needs_db
def test_the_owner_creates_validates_and_seeds_once_and_preserves_existing_rows(target):
    first = target.store()
    recovery = first.try_acquire("llama_control")
    assert recovery.owner_seq == 1
    assert recovery.selection == SelectionRecord("llama_control", None, "active", 1, None, None, None, None)
    assert target.tables() == _TABLES
    first.record_dispatch(_intent())
    first.release()

    second = target.store()
    recovery = second.try_acquire("gemma_candidate")  # a different registry default
    assert recovery.owner_seq == 2
    assert recovery.selection.active_profile_id == "llama_control", "the persisted selection is never re-seeded"
    assert recovery.unresolved_calls == 1 and _calls(target) == [("answer", "dispatching", 1)]
    second.release()


@needs_db
def test_a_same_named_incompatible_table_is_refused_and_left_untouched(target):
    target.query("CREATE TABLE generation_calls (call_id TEXT PRIMARY KEY, note TEXT)")
    target.query("INSERT INTO generation_calls VALUES ('kept', 'an existing row')")
    store = target.store()
    with pytest.raises(SchemaIncompatible):
        store.try_acquire("llama_control")
    assert target.tables() == {"generation_calls"}, "the attempt's DDL rolled back"
    assert target.query("SELECT call_id, note FROM generation_calls") == [("kept", "an existing row")]
    locks = target.query(
        "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' AND classid::bigint = %s AND objid::bigint = %s",
        target.lock_key,
    )
    assert locks == [(0,)], "the refused session was closed, which released its lock"


# --- exclusive ownership -------------------------------------------------------------------


@needs_db
def test_only_one_session_owns_and_ownership_passes_when_that_session_ends(target):
    owner = target.store()
    assert owner.try_acquire("llama_control") is not None
    contender = target.store()
    assert contender.try_acquire("llama_control") is None, "a second process cannot own while the first does"
    assert owner.check_owner()
    owner.release()
    recovery = contender.try_acquire("llama_control")
    assert recovery is not None and recovery.owner_seq == 2
    contender.release()


@needs_db
def test_a_lost_owner_session_can_no_longer_write_and_another_process_takes_over(target):
    owner = target.store()
    owner.try_acquire("llama_control")
    pid = owner._conn.get_backend_pid()
    target.query("SELECT pg_terminate_backend(%s)", (pid,))  # the session is gone (crash, partition, kill)
    with pytest.raises(DispatchIntentRefused):
        owner.record_dispatch(_intent())
    assert not owner.check_owner()
    successor = target.store()
    recovery = successor.try_acquire("llama_control")
    assert recovery is not None and recovery.unresolved_calls == 0, "nothing was recorded by the lost owner"
    successor.release()
    owner.release()


@needs_db
def test_each_fence_refuses_a_stale_owner_on_its_own(target):
    stale = target.store()
    stale.try_acquire("llama_control")
    # 1. The owner row moved on (a newer owner bumped it): the token fence refuses.
    target.query("UPDATE generation_owner SET owner_seq = owner_seq + 1, owner_token = %s::uuid",
                 (str(uuid.uuid4()),))
    with pytest.raises(DispatchIntentRefused):
        stale.record_dispatch(_intent())
    switching = stale.begin_switch(
        expected=SelectionRecord("llama_control", None, "active", 1, None, None, None, None),
        target_profile_id="gemma_candidate", switch_id=str(uuid.uuid4()))
    assert switching is None and not stale.check_owner()
    stale.release()

    # 2. The session gave up its advisory lock but kept the token: the lock fence refuses.
    holder = target.store()
    holder.try_acquire("llama_control")
    holder._conn.cursor().execute("SELECT pg_advisory_unlock(%s, %s)", target.lock_key)
    with pytest.raises(DispatchIntentRefused):
        holder.record_dispatch(_intent())
    assert not holder.check_owner()
    assert _calls(target) == []
    holder.release()


# --- intents, restart recovery ------------------------------------------------------------


@needs_db
def test_intents_are_retired_only_by_an_observed_end_and_survive_a_crash(target):
    owner = target.store()
    owner.try_acquire("llama_control")
    observed, never, unknown = _intent(), _intent(), _intent()
    helper_job = str(uuid.uuid4())
    for intent in (observed, never, unknown):
        owner.record_dispatch(intent)
    owner.record_outcome(observed, CallState.OBSERVED_TERMINAL)
    owner.record_outcome(never, CallState.NEVER_DISPATCHED)
    owner.record_outcome(unknown, CallState.UNKNOWN)
    owner.record_dispatch(_intent("helper", "helper", job_id=helper_job))  # in flight at the crash
    assert sorted(_calls(target)) == [("answer", "unknown", 1), ("helper", "dispatching", 1)]
    owner._conn.close()  # the process dies without reporting anything more

    restarted = target.store()
    recovery = restarted.try_acquire("llama_control")
    assert recovery.unresolved_jobs == {JobKind.ANSWER: 1, JobKind.HELPER: 1}
    assert recovery.unresolved_calls == 2
    assert restarted.count_unresolved() == 2, "startup never clears possibly-running work"
    restarted.release()


# --- the selection's compare-and-set ---------------------------------------------------------


@needs_db
def test_compare_and_set_fences_stale_epochs_and_late_commits(target):
    boot = str(uuid.uuid4())
    owner = target.store(boot)
    owner.try_acquire("llama_control")
    active = owner.read_selection()
    first = owner.begin_switch(expected=active, target_profile_id="gemma_candidate", switch_id=str(uuid.uuid4()))
    assert (first.state, first.epoch, first.switch_owner_boot_id) == ("switching", 2, boot)
    assert owner.begin_switch(expected=active, target_profile_id="gemma_candidate",
                              switch_id=str(uuid.uuid4())) is None, "one switch at a time"
    aborted = owner.abort_switch(switching=first, reason="admin_abort")
    assert (aborted.state, aborted.active_profile_id, aborted.epoch) == ("active", "llama_control", 3)
    assert aborted.last_outcome == "aborted:admin_abort"
    assert owner.commit_switch(switching=first) is None, "the abort's epoch bump fences the late commit"
    assert owner.begin_switch(expected=active, target_profile_id="gemma_candidate",
                              switch_id=str(uuid.uuid4())) is None, "a stale epoch cannot begin"

    second = owner.begin_switch(expected=aborted, target_profile_id="gemma_candidate", switch_id=str(uuid.uuid4()))
    other_boot = SelectionRecord(*([second.active_profile_id, second.target_profile_id, second.state, second.epoch,
                                    second.switch_id, str(uuid.uuid4()), None, None]))
    assert owner.commit_switch(switching=other_boot) is None, "the switch-owner boot id must match too"
    committed = owner.commit_switch(switching=second)
    assert (committed.active_profile_id, committed.state, committed.epoch) == ("gemma_candidate", "active", 5)
    assert committed.last_switch_id == second.switch_id and committed.last_outcome == "committed"
    assert owner.commit_switch(switching=second) is None
    owner.release()


@needs_db
def test_an_interrupted_switch_is_recovered_as_switching_and_only_an_abort_ends_it(target):
    crashed_boot = str(uuid.uuid4())
    crashed = target.store(crashed_boot)
    crashed.try_acquire("llama_control")
    switching = crashed.begin_switch(expected=crashed.read_selection(), target_profile_id="gemma_candidate",
                                     switch_id=str(uuid.uuid4()))
    crashed._conn.close()

    restarted = target.store()
    recovery = restarted.try_acquire("llama_control")
    assert recovery.selection == switching, "startup never resets switching to active"
    assert recovery.selection.switch_owner_boot_id == crashed_boot
    aborted = restarted.abort_switch(switching=recovery.selection, reason="interrupted_switch")
    assert (aborted.state, aborted.active_profile_id, aborted.epoch) == ("active", "llama_control", 3)
    restarted.release()
