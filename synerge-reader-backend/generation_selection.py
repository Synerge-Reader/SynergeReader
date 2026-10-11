"""Durable shared model selection and single-process generation ownership.

PostgreSQL holds three additive tables that belong to this module only:

* ``generation_selection`` -- one row: the shared ACTIVE profile, a TARGET
  while a switch is in progress, the transition STATE (``active`` /
  ``switching``), the EPOCH (bumped by every transition: begin, commit and
  abort) and the boot id of the process that owns an in-progress switch;
* ``generation_owner`` -- one row: the current owner's fencing sequence and
  token, its boot id and a heartbeat;
* ``generation_calls`` -- one row per model call whose end has not been
  observed: written (and committed) immediately BEFORE the model POST, deleted
  only when that call's terminal marker was observed (or when the POST was
  never sent), and otherwise kept -- across restarts -- as unresolved work.

Exclusive ownership. A process owns generation only while it holds the
PostgreSQL session advisory lock ``OWNER_LOCK_KEY`` on its own dedicated owner
session. The lock is held by that session, so it disappears when the session
ends -- a crash, a closed connection, or a terminated backend -- and another
process can then take it. Every write this module makes runs ON that session
and is additionally fenced in the same statement by the owner row's token and
sequence AND by the advisory lock being held by this very backend, so a stale
or replaced owner can neither record a dispatch intent nor move the selection.
A boot id compare-and-set alone is never relied on.

Fail closed. Any failure, including an ambiguous one (a statement whose
outcome the client did not learn), means: no dispatch, no admission change
this module cannot confirm. Startup never clears unresolved calls and never
resets a ``switching`` row to ``active``; it reports them, holds their
capacity, and leaves an interrupted switch for an administrator to abort.

Preconditions (documented, not enforced here): a direct PostgreSQL session
(a transaction-pooling proxy would break session advisory locks), and one
application database per deployment. Nothing here can see or control
embedding models, other Ollama clients or other jobs on a shared host.

No I/O happens at import or construction; ``connect`` is only called by
``PostgresGenerationStore.try_acquire``. Log lines carry fixed categories,
ids and counts only.
"""

from __future__ import annotations

import enum
import re
import threading
import uuid
from dataclasses import dataclass
from typing import Callable, Mapping, Optional

from generation_admission import (
    CallState,
    DispatchIntent,
    DispatchIntents,
    GenerationLedger,
    JobKind,
    SharedSelection,
)

# Two int4 keys ("SYNR", "GEN1"): pg_try_advisory_lock(int4, int4). In
# pg_locks such a lock shows classid = key 1, objid = key 2, objsubid = 2.
OWNER_LOCK_KEY = (0x53594E52, 0x47454E31)
STATEMENT_TIMEOUT_MS = 5000

STATE_ACTIVE = "active"
STATE_SWITCHING = "switching"

# The only values last_outcome may hold.
OUTCOME_COMMITTED = "committed"
ABORT_REASONS = (
    "admin_abort",
    "drain_timeout",
    "uncertain_work",
    "residency_unavailable",
    "residency_timeout",
    "previous_profile_unknown",
    "target_identity_mismatch",
    "target_identity_unavailable",
    "selection_conflict",
    "interrupted_switch",
)

_PROFILE_ID = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


class GenerationStoreError(RuntimeError):
    """A durable selection/ownership operation failed. Fixed messages only."""


class SchemaIncompatible(GenerationStoreError):
    """A same-named object exists with a shape this module does not own."""


class SelectionInvalid(GenerationStoreError):
    """The persisted selection row violates its invariants."""


class OwnershipLost(GenerationStoreError):
    """This store's session no longer holds generation ownership."""


class DispatchIntentRefused(GenerationStoreError):
    """The dispatch intent was not recorded; the call must not be sent."""


# --- schema ----------------------------------------------------------------------

_DDL = (
    """
    CREATE TABLE IF NOT EXISTS generation_selection (
        singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
        active_profile_id TEXT NOT NULL,
        target_profile_id TEXT,
        state TEXT NOT NULL,
        epoch BIGINT NOT NULL CHECK (epoch >= 1),
        switch_id UUID,
        switch_owner_boot_id UUID,
        last_switch_id UUID,
        last_outcome TEXT,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        CHECK (
            (state = 'active' AND target_profile_id IS NULL AND switch_id IS NULL
                AND switch_owner_boot_id IS NULL)
            OR (state = 'switching' AND target_profile_id IS NOT NULL AND switch_id IS NOT NULL
                AND switch_owner_boot_id IS NOT NULL)
        )
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS generation_owner (
        singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
        owner_seq BIGINT NOT NULL CHECK (owner_seq >= 1),
        owner_token UUID NOT NULL,
        boot_id UUID NOT NULL,
        acquired_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        heartbeat_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS generation_calls (
        call_id UUID PRIMARY KEY,
        job_id UUID NOT NULL,
        job_kind TEXT NOT NULL CHECK (job_kind IN ('answer', 'helper')),
        call_role TEXT NOT NULL CHECK (call_role IN ('answer', 'verifier', 'helper')),
        profile_id TEXT NOT NULL,
        model_tag TEXT NOT NULL,
        selection_epoch BIGINT NOT NULL,
        owner_seq BIGINT NOT NULL,
        boot_id UUID NOT NULL,
        state TEXT NOT NULL CHECK (state IN ('dispatching', 'unknown')),
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        resolved_at TIMESTAMPTZ
    )
    """,
)

_TIMESTAMPTZ = "timestamp with time zone"
# table -> {column: (format_type, nullable)}, and its primary key column.
EXPECTED_SCHEMA = {
    "generation_selection": (
        {
            "singleton": ("boolean", False),
            "active_profile_id": ("text", False),
            "target_profile_id": ("text", True),
            "state": ("text", False),
            "epoch": ("bigint", False),
            "switch_id": ("uuid", True),
            "switch_owner_boot_id": ("uuid", True),
            "last_switch_id": ("uuid", True),
            "last_outcome": ("text", True),
            "updated_at": (_TIMESTAMPTZ, False),
        },
        "singleton",
    ),
    "generation_owner": (
        {
            "singleton": ("boolean", False),
            "owner_seq": ("bigint", False),
            "owner_token": ("uuid", False),
            "boot_id": ("uuid", False),
            "acquired_at": (_TIMESTAMPTZ, False),
            "heartbeat_at": (_TIMESTAMPTZ, False),
        },
        "singleton",
    ),
    "generation_calls": (
        {
            "call_id": ("uuid", False),
            "job_id": ("uuid", False),
            "job_kind": ("text", False),
            "call_role": ("text", False),
            "profile_id": ("text", False),
            "model_tag": ("text", False),
            "selection_epoch": ("bigint", False),
            "owner_seq": ("bigint", False),
            "boot_id": ("uuid", False),
            "state": ("text", False),
            "created_at": (_TIMESTAMPTZ, False),
            "resolved_at": (_TIMESTAMPTZ, True),
        },
        "call_id",
    ),
}

_RELKIND_SQL = "SELECT c.relkind FROM pg_class c WHERE c.oid = to_regclass(%s)"
_COLUMNS_SQL = (
    "SELECT a.attname, format_type(a.atttypid, a.atttypmod), NOT a.attnotnull "
    "FROM pg_attribute a WHERE a.attrelid = to_regclass(%s) AND a.attnum > 0 AND NOT a.attisdropped"
)
_PRIMARY_KEY_SQL = (
    "SELECT a.attname FROM pg_index i JOIN pg_attribute a "
    "ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey) "
    "WHERE i.indrelid = to_regclass(%s) AND i.indisprimary"
)


def validate_generation_schema(cursor) -> None:
    """Refuse (SchemaIncompatible) unless each table is an ordinary table with
    exactly the expected columns, types, nullability and primary key. Read
    only; nothing is altered, dropped or rewritten."""
    for table, (columns, primary_key) in EXPECTED_SCHEMA.items():
        cursor.execute(_RELKIND_SQL, (table,))
        row = cursor.fetchone()
        if not row or row[0] != "r":
            raise SchemaIncompatible("a generation table is missing or is not an ordinary table")
        cursor.execute(_COLUMNS_SQL, (table,))
        found = {name: (type_name, bool(nullable)) for name, type_name, nullable in cursor.fetchall()}
        if found != columns:
            raise SchemaIncompatible("a generation table exists with an incompatible shape")
        cursor.execute(_PRIMARY_KEY_SQL, (table,))
        if [r[0] for r in cursor.fetchall()] != [primary_key]:
            raise SchemaIncompatible("a generation table has an incompatible primary key")


# --- values -----------------------------------------------------------------------


@dataclass(frozen=True)
class SelectionRecord:
    active_profile_id: str
    target_profile_id: Optional[str]
    state: str
    epoch: int
    switch_id: Optional[str]
    switch_owner_boot_id: Optional[str]
    last_switch_id: Optional[str]
    last_outcome: Optional[str]

    def __post_init__(self) -> None:
        ok = (
            isinstance(self.active_profile_id, str) and _PROFILE_ID.match(self.active_profile_id)
            and type(self.epoch) is int and self.epoch >= 1
            and self.state in (STATE_ACTIVE, STATE_SWITCHING)
            and all(v is None or (isinstance(v, str) and _UUID.match(v))
                    for v in (self.switch_id, self.switch_owner_boot_id, self.last_switch_id))
            and (self.last_outcome is None or self.last_outcome == OUTCOME_COMMITTED
                 or (isinstance(self.last_outcome, str) and self.last_outcome.startswith("aborted:")
                     and self.last_outcome[len("aborted:"):] in ABORT_REASONS))
        )
        if ok and self.state == STATE_ACTIVE:
            ok = self.target_profile_id is None and self.switch_id is None and self.switch_owner_boot_id is None
        elif ok:
            ok = (
                isinstance(self.target_profile_id, str) and bool(_PROFILE_ID.match(self.target_profile_id))
                and self.switch_id is not None and self.switch_owner_boot_id is not None
            )
        if not ok:
            raise SelectionInvalid("the persisted generation selection is invalid")

    @property
    def shared(self) -> SharedSelection:
        return SharedSelection(self.active_profile_id, self.epoch)


@dataclass(frozen=True)
class OwnerRecovery:
    """What a new owner found: the persisted selection and the unresolved
    calls earlier owners left (counted as one job each)."""

    owner_seq: int
    selection: SelectionRecord
    unresolved_jobs: Mapping[JobKind, int]
    unresolved_calls: int


_SELECTION_COLUMNS = (
    "active_profile_id, target_profile_id, state, epoch, switch_id::text, "
    "switch_owner_boot_id::text, last_switch_id::text, last_outcome"
)


def _selection_from_row(row) -> SelectionRecord:
    if not row or len(row) != 8:
        raise SelectionInvalid("the persisted generation selection is missing")
    return SelectionRecord(*row)


# --- the PostgreSQL store ---------------------------------------------------------------


class PostgresGenerationStore(DispatchIntents):
    """The owner session and every durable operation on it.

    One instance per acquisition attempt. ``try_acquire`` opens the dedicated
    session (via ``connect``), takes the advisory lock without waiting,
    creates or validates the tables in one transaction, seeds the selection
    row only if it does not exist, bumps the owner sequence and returns what
    it recovered. All later statements run on that session, serialized by a
    lock, in autocommit mode, each bounded server-side by
    ``statement_timeout_ms``.
    """

    def __init__(
        self,
        connect: Callable[[], object],
        *,
        boot_id: str,
        lock_key=OWNER_LOCK_KEY,
        statement_timeout_ms: int = STATEMENT_TIMEOUT_MS,
    ) -> None:
        if not isinstance(boot_id, str) or not _UUID.match(boot_id):
            raise ValueError("boot_id must be a canonical UUID")
        key_ok = (
            isinstance(lock_key, tuple) and len(lock_key) == 2
            and all(type(k) is int and -2**31 <= k < 2**31 for k in lock_key)
        )
        if not key_ok:
            raise ValueError("lock_key is two int4 values")
        if type(statement_timeout_ms) is not int or not 100 <= statement_timeout_ms <= 60000:
            raise ValueError("statement_timeout_ms must be an integer in [100, 60000]")
        self._connect = connect
        self.boot_id = boot_id
        self._lock_key = lock_key
        self._timeout_ms = statement_timeout_ms
        self._lock = threading.Lock()
        self._conn = None
        self._owner_seq: Optional[int] = None
        self._owner_token: Optional[str] = None

    # --- ownership ---------------------------------------------------------------------

    @property
    def owner_seq(self) -> Optional[int]:
        return self._owner_seq

    def _fence_params(self):
        return (self._owner_token, self._owner_seq, self._lock_key[0], self._lock_key[1])

    # Appended to every write: this session's owner row AND its advisory lock.
    _FENCE = (
        "EXISTS (SELECT 1 FROM generation_owner o WHERE o.singleton AND o.owner_token = %s::uuid "
        "AND o.owner_seq = %s) AND EXISTS (SELECT 1 FROM pg_locks l WHERE l.locktype = 'advisory' "
        "AND l.classid::bigint = %s AND l.objid::bigint = %s AND l.objsubid = 2 "
        "AND l.pid = pg_backend_pid() AND l.granted)"
    )

    def try_acquire(self, default_profile_id: str) -> Optional[OwnerRecovery]:
        """OwnerRecovery when this store became the owner; None when another
        session holds ownership; raises on schema, selection or database
        failure (the session is then closed, which releases any lock)."""
        if not isinstance(default_profile_id, str) or not _PROFILE_ID.match(default_profile_id):
            raise ValueError("default_profile_id is a well-formed profile id")
        with self._lock:
            if self._conn is not None:
                raise GenerationStoreError("this store already owns a session")
            conn = self._connect()
            if conn is None:
                raise GenerationStoreError("the owner session could not be opened")
            try:
                recovery = self._acquire_on(conn, default_profile_id)
            except BaseException:
                _close_quietly(conn)
                raise
            if recovery is None:
                _close_quietly(conn)
                return None
            self._conn = conn
            return recovery

    def _acquire_on(self, conn, default_profile_id: str) -> Optional[OwnerRecovery]:
        # The application connector leaves a transaction open (its vector
        # registration query): end it before changing the session mode.
        conn.rollback()
        conn.autocommit = True
        cursor = conn.cursor()
        cursor.execute("SET statement_timeout = %s", (self._timeout_ms,))
        cursor.execute("SELECT pg_try_advisory_lock(%s, %s)", self._lock_key)
        row = cursor.fetchone()
        if not row or row[0] is not True:
            return None
        # Owner from here on: DDL, validation, seed and the owner bump commit
        # together or not at all.
        token = str(uuid.uuid4())
        conn.autocommit = False
        try:
            for statement in _DDL:
                cursor.execute(statement)
            validate_generation_schema(cursor)
            cursor.execute(
                "INSERT INTO generation_selection (singleton, active_profile_id, state, epoch) "
                "VALUES (TRUE, %s, 'active', 1) ON CONFLICT (singleton) DO NOTHING",
                (default_profile_id,),
            )
            cursor.execute(
                "INSERT INTO generation_owner (singleton, owner_seq, owner_token, boot_id) "
                "VALUES (TRUE, 1, %s::uuid, %s::uuid) ON CONFLICT (singleton) DO UPDATE SET "
                "owner_seq = generation_owner.owner_seq + 1, owner_token = EXCLUDED.owner_token, "
                "boot_id = EXCLUDED.boot_id, acquired_at = now(), heartbeat_at = now() RETURNING owner_seq",
                (token, self.boot_id),
            )
            owner_seq = cursor.fetchone()[0]
            cursor.execute(f"SELECT {_SELECTION_COLUMNS} FROM generation_selection WHERE singleton")
            selection = _selection_from_row(cursor.fetchone())
            cursor.execute(
                "SELECT job_kind, count(DISTINCT job_id), count(*) FROM generation_calls GROUP BY job_kind"
            )
            jobs = {JobKind.ANSWER: 0, JobKind.HELPER: 0}
            calls = 0
            for kind, job_count, call_count in cursor.fetchall():
                jobs[JobKind(kind)] = int(job_count)
                calls += int(call_count)
            conn.commit()
        except BaseException:
            try:
                conn.rollback()
            except Exception:
                pass
            raise
        conn.autocommit = True
        self._owner_seq = int(owner_seq)
        self._owner_token = token
        return OwnerRecovery(owner_seq=int(owner_seq), selection=selection, unresolved_jobs=jobs,
                             unresolved_calls=calls)

    def check_owner(self) -> bool:
        """Heartbeat: True only if this session still holds the advisory lock
        and the owner row is still this store's. Any failure is False."""
        try:
            row = self._one(
                "UPDATE generation_owner o SET heartbeat_at = now() WHERE o.singleton AND "
                + self._FENCE + " RETURNING o.owner_seq",
                self._fence_params(),
            )
        except Exception:
            return False
        return bool(row) and row[0] == self._owner_seq

    def release(self) -> None:
        """Give up ownership: unlock (best effort) and close the session,
        which releases the lock in any case. Idempotent."""
        with self._lock:
            conn, self._conn = self._conn, None
            self._owner_token = None
        if conn is None:
            return
        try:
            conn.cursor().execute("SELECT pg_advisory_unlock(%s, %s)", self._lock_key)
        except Exception:
            pass
        _close_quietly(conn)

    def _one(self, sql: str, params=()):
        with self._lock:
            conn = self._conn
            if conn is None or self._owner_token is None:
                raise OwnershipLost("no owner session")
            cursor = conn.cursor()
            try:
                cursor.execute(sql, params)
                return cursor.fetchone() if cursor.description else None
            finally:
                try:
                    cursor.close()
                except Exception:
                    pass

    def _all(self, sql: str, params=()):
        with self._lock:
            conn = self._conn
            if conn is None or self._owner_token is None:
                raise OwnershipLost("no owner session")
            cursor = conn.cursor()
            try:
                cursor.execute(sql, params)
                return cursor.fetchall()
            finally:
                try:
                    cursor.close()
                except Exception:
                    pass

    # --- dispatch intents --------------------------------------------------------------

    def record_dispatch(self, intent: DispatchIntent) -> None:
        if not isinstance(intent, DispatchIntent):
            raise DispatchIntentRefused("not a dispatch intent")
        try:
            row = self._one(
                "INSERT INTO generation_calls (call_id, job_id, job_kind, call_role, profile_id, model_tag, "
                "selection_epoch, owner_seq, boot_id, state) "
                "SELECT %s::uuid, %s::uuid, %s, %s, %s, %s, %s, %s, %s::uuid, 'dispatching' WHERE "
                + self._FENCE + " RETURNING call_id::text",
                (intent.call_id, intent.job_id, intent.job_kind, intent.call_role, intent.profile_id,
                 intent.model_tag, intent.selection_epoch, self._owner_seq, self.boot_id,
                 *self._fence_params()),
            )
        except GenerationStoreError:
            raise
        except Exception:
            raise DispatchIntentRefused("the dispatch intent could not be recorded") from None
        if not row or row[0] != intent.call_id:
            raise DispatchIntentRefused("this process is not the generation owner")

    def record_outcome(self, intent: DispatchIntent, state: CallState) -> None:
        if state in (CallState.OBSERVED_TERMINAL, CallState.NEVER_DISPATCHED):
            sql = "DELETE FROM generation_calls WHERE call_id = %s::uuid AND " + self._FENCE
        elif state is CallState.UNKNOWN:
            sql = ("UPDATE generation_calls SET state = 'unknown', resolved_at = now() "
                   "WHERE call_id = %s::uuid AND state = 'dispatching' AND " + self._FENCE)
        else:
            raise ValueError("unknown call state")
        self._one(sql, (intent.call_id, *self._fence_params()))

    def count_unresolved(self) -> int:
        row = self._one("SELECT count(*) FROM generation_calls")
        return int(row[0])

    # --- selection ---------------------------------------------------------------------

    def read_selection(self) -> SelectionRecord:
        return _selection_from_row(
            self._one(f"SELECT {_SELECTION_COLUMNS} FROM generation_selection WHERE singleton")
        )

    def begin_switch(self, *, expected: SelectionRecord, target_profile_id: str,
                     switch_id: str) -> Optional[SelectionRecord]:
        """active -> switching, only from exactly ``expected`` (state, active
        profile and epoch). None when the CAS did not match."""
        row = self._one(
            "UPDATE generation_selection SET state = 'switching', target_profile_id = %s, "
            "switch_id = %s::uuid, switch_owner_boot_id = %s::uuid, epoch = epoch + 1, updated_at = now() "
            "WHERE singleton AND state = 'active' AND active_profile_id = %s AND epoch = %s AND "
            + self._FENCE + f" RETURNING {_SELECTION_COLUMNS}",
            (target_profile_id, switch_id, self.boot_id, expected.active_profile_id, expected.epoch,
             *self._fence_params()),
        )
        return _selection_from_row(row) if row else None

    def commit_switch(self, *, switching: SelectionRecord) -> Optional[SelectionRecord]:
        """switching -> active with the target, only if state, target, epoch,
        switch id AND switch-owner boot id all still match ``switching``."""
        row = self._one(
            "UPDATE generation_selection SET active_profile_id = target_profile_id, target_profile_id = NULL, "
            "state = 'active', epoch = epoch + 1, switch_id = NULL, switch_owner_boot_id = NULL, "
            "last_switch_id = switch_id, last_outcome = 'committed', updated_at = now() "
            "WHERE singleton AND state = 'switching' AND target_profile_id = %s AND epoch = %s "
            "AND switch_id = %s::uuid AND switch_owner_boot_id = %s::uuid AND "
            + self._FENCE + f" RETURNING {_SELECTION_COLUMNS}",
            (switching.target_profile_id, switching.epoch, switching.switch_id,
             switching.switch_owner_boot_id, *self._fence_params()),
        )
        return _selection_from_row(row) if row else None

    def abort_switch(self, *, switching: SelectionRecord, reason: str) -> Optional[SelectionRecord]:
        """switching -> active with the UNCHANGED active profile, only if the
        state, switch id and epoch still match. The epoch bump fences any late
        commit of the aborted switch."""
        if reason not in ABORT_REASONS:
            raise ValueError("unknown abort reason")
        row = self._one(
            "UPDATE generation_selection SET target_profile_id = NULL, state = 'active', epoch = epoch + 1, "
            "switch_id = NULL, switch_owner_boot_id = NULL, last_switch_id = switch_id, last_outcome = %s, "
            "updated_at = now() WHERE singleton AND state = 'switching' AND switch_id = %s::uuid "
            "AND epoch = %s AND " + self._FENCE + f" RETURNING {_SELECTION_COLUMNS}",
            ("aborted:" + reason, switching.switch_id, switching.epoch, *self._fence_params()),
        )
        return _selection_from_row(row) if row else None


def _close_quietly(conn) -> None:
    try:
        conn.close()
    except Exception:
        pass


# --- the owner manager ------------------------------------------------------------------


class OwnerState(enum.Enum):
    NOT_STARTED = "not_started"
    ACQUIRING = "acquiring"
    WAITING_FOR_LOCK = "waiting_for_lock"
    STORE_UNAVAILABLE = "store_unavailable"
    SCHEMA_INCOMPATIBLE = "schema_incompatible"
    OWNER = "owner"
    LOST = "lost"
    STOPPED = "stopped"


@dataclass(frozen=True)
class OwnerSettings:
    """Provisional, unmeasured: how often a non-owner retries, and how often
    the owner proves it still holds the lock."""

    retry_s: float = 5.0
    heartbeat_s: float = 5.0

    def __post_init__(self) -> None:
        for name in ("retry_s", "heartbeat_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.01 <= value <= 300:
                raise ValueError(f"{name} must be a number in [0.01, 300]")


class GenerationOwner:
    """Acquires and holds this process's generation ownership on one
    dedicated thread.

    The thread retries until it owns (another process may still hold the lock
    during an overlapping restart), installs the recovered state into the
    ledger, then heartbeats. Ownership is taken at most once per process: a
    lost or stopped owner fences the ledger for good -- no admission and no
    dispatch -- and never re-acquires; a restart is the recovery path.
    ``start`` and ``stop`` never wait for the thread.
    """

    def __init__(
        self,
        *,
        store_factory: Callable[[], PostgresGenerationStore],
        ledger: GenerationLedger,
        default_profile_id: Callable[[], str],
        settings: OwnerSettings = OwnerSettings(),
        thread_factory: Callable[..., object] = threading.Thread,
        log: Callable[[str], None] = print,
    ) -> None:
        self._store_factory = store_factory
        self._ledger = ledger
        self._default_profile_id = default_profile_id
        self._settings = settings
        self._thread_factory = thread_factory
        self._log = log
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._state = OwnerState.NOT_STARTED
        self._started = False
        self._store: Optional[PostgresGenerationStore] = None
        self._recovery: Optional[OwnerRecovery] = None

    @property
    def state(self) -> OwnerState:
        with self._lock:
            return self._state

    @property
    def store(self) -> Optional[PostgresGenerationStore]:
        """The owning store while (and only while) this process owns."""
        with self._lock:
            return self._store if self._state is OwnerState.OWNER else None

    @property
    def recovery(self) -> Optional[OwnerRecovery]:
        with self._lock:
            return self._recovery

    def start(self) -> bool:
        with self._lock:
            if self._started:
                return False
            self._started = True
            self._state = OwnerState.ACQUIRING
        try:
            thread = self._thread_factory(target=self._run, name="generation-owner", daemon=True)
            thread.start()
        except Exception:
            self._set(OwnerState.STOPPED)
            self._ledger.fence()
            self._log("[Generation] ownership thread could not start; generation unavailable")
            return False
        return True

    def stop(self) -> None:
        """Fence the ledger at once and let the thread release ownership."""
        with self._lock:
            self._started = True  # a stopped owner is never started again
        self._ledger.fence()
        self._stop.set()

    def _set(self, state: OwnerState, store=None, recovery=None) -> None:
        with self._lock:
            self._state = state
            self._store = store
            if recovery is not None:
                self._recovery = recovery

    def _run(self) -> None:
        store = None
        final = OwnerState.STOPPED
        reported = None
        try:
            while not self._stop.is_set():
                store = self._store_factory()
                try:
                    recovery = store.try_acquire(self._default_profile_id())
                except (SchemaIncompatible, SelectionInvalid):
                    recovery, waiting = None, OwnerState.SCHEMA_INCOMPATIBLE
                except Exception:
                    recovery, waiting = None, OwnerState.STORE_UNAVAILABLE
                else:
                    waiting = OwnerState.WAITING_FOR_LOCK
                if recovery is None:
                    store.release()
                    store = None
                    self._set(waiting)
                    if reported is not waiting:
                        self._log(f"[Generation] not the generation owner yet ({waiting.value}); retrying")
                        reported = waiting
                    self._stop.wait(self._settings.retry_s)
                    continue
                record = recovery.selection
                installed = self._ledger.install_owner(
                    intents=store,
                    owner_seq=recovery.owner_seq,
                    selection=record.shared,
                    paused_by=record.switch_id if record.state == STATE_SWITCHING else None,
                    recovered=recovery.unresolved_jobs,
                )
                if not installed:
                    return
                self._set(OwnerState.OWNER, store, recovery)
                self._log(
                    f"[Generation] ownership acquired: owner_seq={recovery.owner_seq} "
                    f"epoch={record.epoch} state={record.state} "
                    f"unresolved_answers={recovery.unresolved_jobs[JobKind.ANSWER]} "
                    f"unresolved_helpers={recovery.unresolved_jobs[JobKind.HELPER]}"
                )
                while not self._stop.wait(self._settings.heartbeat_s):
                    if not store.check_owner():
                        final = OwnerState.LOST
                        self._log("[Generation] ownership lost; generation fenced until restart")
                        return
                return
        except Exception:
            self._log("[Generation] ownership thread failed; generation fenced until restart")
        finally:
            self._ledger.fence()
            with self._lock:
                self._store = None
                self._state = final
            if store is not None:
                store.release()
