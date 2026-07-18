# Patch notes for v0.3.28

This patch focuses on two issues from the supplied debug bundle:

1. Dialogue-window sentence fragments being split into separate observations / render boxes.
2. Rendered font size collapsing too aggressively when English output is longer than the source line.

## What changed

### Dialogue merge improvements
- Added same-row dialogue fragment detection so split OCR records on the same visual text row can stay together.
- Added row-aware fragment joining so merged dialogue keeps inline fragments on the same line instead of inserting artificial newlines.
- Extended dialogue-box observation merge and render-cluster merge so same-row fragments and vertically chained continuation lines can be merged more reliably.

### Font sizing / layout consistency
- Added a layout planning pass that measures wrapped text at the preferred source-derived pixel size.
- Dialogue and name renders can now expand their text box height before shrinking font size, which keeps rendered text closer to the original source size.
- Added extra dialogue/name margins and configurable max wrap-height ratios so long English lines wrap instead of collapsing as much.

## Verification performed here
- Python syntax compilation for the patched source files.
- Geometry sanity-check against the supplied `F000126` debug manifest to verify that:
  - the same-row prefix fragment (`“オーラ”`) is treated as part of the same dialogue row as the main sentence,
  - the continuation line below remains chain-mergeable,
  - the row joiner reconstructs the line as a single first-row sentence fragment plus the second-row continuation.

## Not fully re-run in this environment
- Full live OCR + Qt overlay runtime.
- Local llama model execution.
- Windows-only capture / overlay behavior.

Those pieces were omitted from the upload or are not available in this environment, so this patch is based on static analysis plus targeted verification against the provided debug artifacts.
