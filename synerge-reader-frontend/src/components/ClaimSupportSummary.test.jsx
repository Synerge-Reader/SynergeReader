import { render, screen, fireEvent } from "@testing-library/react";
import { ClaimSupportSummary } from "./ClaimSupportSummary";

// Claims shaped exactly as the backend's verification event sends them.
const claim = (status, reason, citationIds = ["C1"], text = "The term is two years [C1].") => ({
  text, citation_ids: citationIds, invalid_citation_ids: [], status, reason,
});
const supported          = () => claim("supported", "verified");
const partial            = () => claim("partially_supported", "verified");
const floorPartial       = () => claim("partially_supported", "evidence_terms_missing");
const checkedUnsupported = () => claim("unsupported", "verified");
const mismatch           = () => claim("unsupported", "evidence_terms_missing");
const uncited            = () => claim("unsupported", "no_citation", [], "IRCoT and Step-Back differ.");
const invalidCitation    = () => claim("unsupported", "invalid_citation", [], "Bad marker [C9].");
const notChecked         = () => claim("unverified", "not_verified");
const unavailable        = () => claim("unverified", "verifier_unavailable");
const malformed          = () => claim("unverified", "verifier_malformed");

const GREEN = "rgb(21, 128, 61)";
const AMBER = "rgb(180, 83, 9)";
const RED = "rgb(185, 28, 28)";
const GREY = "rgb(71, 85, 105)";

function renderSummary(claims) {
  return render(<ClaimSupportSummary claims={claims} citations={[]} onOpen={() => {}} />);
}

function headline() {
  return screen.getByRole("button");
}

function openDetails() {
  fireEvent.click(headline());
}

test("partial-only checks are never shown as supported", () => {
  renderSummary([partial(), floorPartial()]);
  expect(headline()).toHaveTextContent("Partly supported by your documents");
  expect(headline()).not.toHaveTextContent(/^Supported by your documents/);
  expect(headline().style.color).toBe(AMBER);
  expect(screen.getByText("2 of 2 cited claims checked (2 partly supported)")).toBeInTheDocument();
});

test("supported plus partly supported is still not green", () => {
  renderSummary([supported(), partial()]);
  expect(headline()).toHaveTextContent("Partly supported by your documents");
  expect(headline().style.color).toBe(AMBER);
});

test("only fully checked, fully supported answers are green", () => {
  renderSummary([supported(), supported()]);
  expect(headline()).toHaveTextContent("Supported by your documents");
  expect(headline().style.color).toBe(GREEN);
  expect(screen.getByText("2 of 2 cited claims checked (2 supported)")).toBeInTheDocument();
});

test("supported checks with unchecked eligible claims do not imply full verification", () => {
  renderSummary([supported(), supported(), notChecked()]);
  expect(headline()).toHaveTextContent("Citation check incomplete");
  expect(headline().style.color).toBe(GREY);
  expect(screen.getByText("2 of 3 cited claims checked (2 supported) · 1 not checked")).toBeInTheDocument();
});

test("an unsupported verdict is a completed check with its own label", () => {
  renderSummary([checkedUnsupported()]);
  expect(headline()).toHaveTextContent("Parts of this answer are not supported by your documents");
  expect(headline().style.color).toBe(RED);
  expect(screen.getByText("1 of 1 cited claim checked (1 not supported)")).toBeInTheDocument();
  openDetails();
  expect(screen.getByText("Cited source does not support this")).toBeInTheDocument();
  expect(screen.queryByText("Not supported by a cited document")).not.toBeInTheDocument();
});

test("a mixed answer counts unsupported verdicts as checked, not supported", () => {
  renderSummary([supported(), checkedUnsupported(), uncited()]);
  expect(headline()).toHaveTextContent("Some claims in this answer need checking");
  expect(screen.getByText(
    "2 of 2 cited claims checked (1 supported, 1 not supported) · 1 without a usable citation"
  )).toBeInTheDocument();
  openDetails();
  expect(screen.getByText("Supported")).toBeInTheDocument();
  expect(screen.getByText("Cited source does not support this")).toBeInTheDocument();
  expect(screen.getByText("Not supported by a cited document")).toBeInTheDocument();
});

test("uncited, unresolvable and mismatched claims keep their distinct labels", () => {
  renderSummary([uncited(), invalidCitation(), mismatch()]);
  expect(headline()).toHaveTextContent("Parts of this answer are not supported by your documents");
  openDetails();
  expect(screen.getByText("Not supported by a cited document")).toBeInTheDocument();
  expect(screen.getByText("Citation could not be resolved")).toBeInTheDocument();
  expect(screen.getByText("Cited source does not state this")).toBeInTheDocument();
});

test("a failed or malformed verifier means the check is unavailable, never passed", () => {
  renderSummary([supported(), unavailable(), malformed()]);
  expect(headline()).toHaveTextContent("Citation check unavailable");
  expect(headline().style.color).toBe(GREY);
  expect(screen.getByText("1 of 3 cited claims checked (1 supported) · 2 not checked")).toBeInTheDocument();
  openDetails();
  expect(screen.getAllByText("Check unavailable")).toHaveLength(2);
});

test("budget-skipped claims read as not checked", () => {
  renderSummary([notChecked()]);
  expect(headline()).toHaveTextContent("Citation check incomplete");
  openDetails();
  expect(screen.getByText("Not checked")).toBeInTheDocument();
});

test("an empty verification is reported as unavailable, not as a pass", () => {
  renderSummary([]);
  expect(headline()).toHaveTextContent("Citation check unavailable");
  expect(headline()).not.toHaveTextContent("details");
  expect(screen.getByText("No claims were checked")).toBeInTheDocument();
});

test("no verification event renders nothing", () => {
  const { container } = renderSummary(undefined);
  expect(container).toBeEmptyDOMElement();
});

test("unrecognised claim data gives unknown coverage and no green", () => {
  renderSummary([claim("supported", "some_future_reason")]);
  expect(headline()).toHaveTextContent("Citation check incomplete");
  expect(screen.getByText("Check coverage unknown")).toBeInTheDocument();
});

test("claim text goes through the caller's rewriter, and the fallback never prints ids", () => {
  const Rewriter = ({ text }) => <span>rewritten:{text.length}</span>;
  const { unmount } = render(
    <ClaimSupportSummary claims={[supported()]} citations={[]} onOpen={() => {}} ClaimText={Rewriter} />
  );
  openDetails();
  expect(screen.getByText("rewritten:27")).toBeInTheDocument();
  unmount();

  renderSummary([supported()]);
  openDetails();
  expect(screen.getByText("The term is two years.")).toBeInTheDocument();
  expect(screen.queryByText(/\[C1\]/)).not.toBeInTheDocument();
});
