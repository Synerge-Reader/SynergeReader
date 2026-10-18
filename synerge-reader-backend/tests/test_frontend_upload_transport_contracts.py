"""Source-level contracts: the frontend must upload the ORIGINAL file.

These are text/source-shape contract tests. The two JavaScript sources are read
as plain text; nothing here bundles, transpiles, imports, renders, or executes
any frontend code, and no npm, browser, network, database, or Ollama access
occurs.

The defect these tests exist to prevent is specific and has already shipped
once: the active upload path built ``new Blob([parsed.text], {type:"text/plain"})``
and uploaded that text blob under the original filename, destroying the PDF/DOCX
bytes that server-side ingestion needs. The browser may still parse a file
locally for preview, page display, text selection, and suggested questions --
but that parse must never become the bytes sent to /upload.

Scope is deliberately narrow: only the upload block of each file is inspected,
so unrelated ``new Blob(...)`` uses elsewhere (CSV/JSONL export, transcript
download) stay legal. The final test in this file pins that narrowness down by
asserting one of those unrelated uses is still present and still passing.

GridApp's transport lives in ``src/uploadJobs.js`` (``uploadAndWait``): it posts
the original File for background processing and polls the job for the same
batch envelope. GridApp's processFiles block hands it the File object and keeps
the "indexed only" acceptance rule, so both pieces are inspected here.

What these tests do NOT prove: they do not prove the upload works at runtime,
that the response renders correctly, or that the local preview parses anything.
"""

from pathlib import Path

import pytest


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_GRID_APP_PATH = _REPOSITORY_ROOT / "synerge-reader-frontend" / "src" / "GridApp.jsx"
_UPLOAD_JOBS_PATH = _REPOSITORY_ROOT / "synerge-reader-frontend" / "src" / "uploadJobs.js"
_FILE_UPLOAD_PATH = (
    _REPOSITORY_ROOT
    / "synerge-reader-frontend"
    / "src"
    / "components"
    / "FileUpload.js"
)


def _read(path):
    assert path.is_file(), f"expected {path} to exist"
    return path.read_text(encoding="utf-8")


def _slice_between(source, start_marker, end_marker, *, label):
    start = source.find(start_marker)
    assert start != -1, f"could not locate {start_marker!r} in {label}"
    end = source.find(end_marker, start)
    assert end != -1, f"could not locate {end_marker!r} after {start_marker!r} in {label}"
    return source[start:end]


@pytest.fixture(scope="module")
def grid_app_source():
    return _read(_GRID_APP_PATH)


@pytest.fixture(scope="module")
def file_upload_source():
    return _read(_FILE_UPLOAD_PATH)


@pytest.fixture(scope="module")
def grid_app_upload_block(grid_app_source):
    """Just the active upload path: the processFiles body up to its suggested
    questions. Everything outside this window is out of scope."""
    block = _slice_between(
        grid_app_source,
        "const processFiles = useCallback",
        "// Generate suggested questions",
        label="GridApp.jsx",
    )
    assert "await uploadAndWait({" in block, (
        "the inspected GridApp block must be the one that hands the file to the upload transport"
    )
    return block


@pytest.fixture(scope="module")
def upload_transport_block():
    """GridApp's upload transport: uploadAndWait in uploadJobs.js."""
    block = _slice_between(
        _read(_UPLOAD_JOBS_PATH),
        "export async function uploadAndWait",
        "onStatus(describeJob(job));",
        label="uploadJobs.js",
    )
    assert "${backend}/upload`" in block, (
        "the inspected transport block must be the one that POSTs to /upload"
    )
    return block


@pytest.fixture(scope="module")
def file_upload_block(file_upload_source):
    """Just FileUpload.js's dormant/legacy batch upload helper."""
    block = _slice_between(
        file_upload_source,
        "const uploadBatchToBackend",
        "const processPDF",
        label="FileUpload.js",
    )
    assert "'/upload'" in block, (
        "the inspected FileUpload.js block must be the one that POSTs to /upload"
    )
    return block


# --- the active GridApp path sends the original File -------------------------


def test_grid_app_uploads_the_original_file_object(grid_app_upload_block, upload_transport_block):
    assert "          file,\n" in grid_app_upload_block, (
        "GridApp must hand the original File object to the upload transport"
    )
    assert 'fd.append("files", file, file.name)' in upload_transport_block, (
        "the transport must append the original File object under its own name, so the "
        "backend receives the original PDF/DOCX/TXT bytes"
    )


def test_grid_app_upload_block_constructs_no_blob(grid_app_upload_block, upload_transport_block):
    for block in (grid_app_upload_block, upload_transport_block):
        assert "new Blob(" not in block, (
            "the GridApp upload path must not construct any Blob -- a Blob here "
            "replaces the original file bytes with client-extracted text"
        )


def test_grid_app_never_uploads_a_parsed_text_blob_anywhere(grid_app_source):
    for banned in ("new Blob([parsed.text]", "new Blob([parsed.text],"):
        assert banned not in grid_app_source, (
            f"{banned!r} is the exact defect this slice removes: it uploads "
            "client-extracted text in place of the original file"
        )


def test_grid_app_still_parses_locally_for_preview(grid_app_upload_block):
    # Local parsing stays -- it feeds the preview, page display, selection and
    # suggestions. It simply must not be the transport.
    assert "await parsePDF(file)" in grid_app_upload_block
    assert "await parseDOCX(file)" in grid_app_upload_block
    assert "await parseTXT(file)" in grid_app_upload_block


def test_grid_app_sends_auth_token_when_present(grid_app_upload_block, upload_transport_block):
    assert "          authToken,\n" in grid_app_upload_block
    assert 'if (authToken) fd.append("auth_token", authToken)' in upload_transport_block
    assert "Authorization: `Bearer ${authToken" in upload_transport_block, (
        "job polling must identify the caller with the bearer token"
    )


# --- the active GridApp path reads the batch envelope truthfully ------------


def test_grid_app_reads_the_batch_envelope_results(upload_transport_block):
    assert "Array.isArray(data.results)" in upload_transport_block, (
        "the transport must read the batch envelope's results array, not the old "
        "bare-list response shape"
    )
    assert "Array.isArray(envelope.results)" in upload_transport_block, (
        "a finished background job must be read through the same batch envelope"
    )


def test_grid_app_reads_non_2xx_envelopes_too(upload_transport_block):
    assert "res.json()" in upload_transport_block
    assert "if (res.ok)" not in upload_transport_block, (
        "422 and 503 responses still carry a structured envelope, so the body "
        "must not be gated behind res.ok"
    )


def test_grid_app_admits_only_indexed_documents_to_the_ui(grid_app_upload_block):
    assert 'result.status !== "indexed"' in grid_app_upload_block, (
        "a document may enter the UI only when the backend reports it indexed"
    )
    assert "!result.document_id" in grid_app_upload_block, (
        "a usable backend document_id is required before the document is shown"
    )
    assert "continue;" in grid_app_upload_block, (
        "a rejected or failed file must be skipped, not added to the UI"
    )


def test_grid_app_shows_the_contract_safe_error_message(grid_app_upload_block):
    assert "result?.error_message" in grid_app_upload_block, (
        "rejected/failed files must surface the ingestion contract's safe "
        "error_message"
    )


def test_grid_app_uses_backend_supplied_result_fields(grid_app_upload_block):
    for field in ("result.document_id", "result.filename", "result.chunks_count", "result.warnings"):
        assert field in grid_app_upload_block, (
            f"the UI must use the backend-supplied {field}"
        )


def test_grid_app_does_not_fabricate_a_local_document_id(grid_app_upload_block):
    assert "docId || Date.now()" not in grid_app_upload_block, (
        "a locally invented id would claim a document the backend never stored"
    )


def test_grid_app_keeps_the_users_original_file_name(grid_app_upload_block):
    assert "name:              file.name," in grid_app_upload_block, (
        "the document must keep the name the user recognises"
    )


def test_table_extraction_control_is_behind_an_explicit_build_flag(grid_app_source, grid_app_upload_block):
    # Tables stay unavailable in a public build until their qualification passes.
    assert 'const TABLE_EXTRACTION_ENABLED = process.env.REACT_APP_ENABLE_TABLE_EXTRACTION === "true";' in grid_app_source
    assert 'extraction: isPdf && extractTables && TABLE_EXTRACTION_ENABLED ? "enhanced" : "standard",' in grid_app_upload_block
    checkbox = grid_app_source.index('data-testid="extract-tables-toggle"')
    gate = grid_app_source.rfind("{TABLE_EXTRACTION_ENABLED && (", 0, checkbox)
    assert gate != -1 and checkbox - gate < 800, "the checkbox renders only inside the flag"


# --- the dormant FileUpload.js path cannot reintroduce the defect -----------


def test_file_upload_sends_the_original_file_object(file_upload_block):
    assert "formData.append('files', file, file.name)" in file_upload_block, (
        "the legacy upload helper must append original File objects"
    )


def test_file_upload_block_constructs_no_blob(file_upload_block):
    assert "new Blob(" not in file_upload_block, (
        "the legacy upload helper must not rebuild a text blob, or reviving this "
        "dormant path would reintroduce text-blob transport"
    )


def test_file_upload_never_uploads_a_text_blob_anywhere(file_upload_source):
    assert "new Blob([text]" not in file_upload_source
    assert "new Blob([textContent]" not in file_upload_source


def test_file_upload_reads_the_batch_envelope(file_upload_block):
    assert "Array.isArray(payload.results)" in file_upload_block, (
        "the legacy helper must read the batch envelope, not the old bare list"
    )


def test_file_upload_reports_only_contract_safe_errors(file_upload_source):
    assert "result.error_message" in file_upload_source, (
        "non-indexed files must be reported with the contract's safe message"
    )
    assert "result.error}" not in file_upload_source, (
        "the old free-form per-file 'error' string is gone from the contract"
    )


# --- narrowness: unrelated Blob usage elsewhere stays legal -----------------


def test_unrelated_blob_usage_elsewhere_is_untouched(grid_app_source):
    assert "new Blob([jsonl]" in grid_app_source, (
        "the JSONL export's Blob must still be present -- if this fails, either "
        "an unrelated feature was changed or these checks are no longer narrow"
    )
    assert "new Blob([content]" in grid_app_source, (
        "the transcript download's Blob must still be present"
    )
