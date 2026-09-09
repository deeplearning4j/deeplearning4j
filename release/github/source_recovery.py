"""Explicitly authorized compiled Java repair; never a documentation overlay.

Only ProgressInputStream.afterRead's checked-exception declaration is eligible.
Native receipts, packaging provenance and documentation provenance stay separate.
"""
import hashlib
import json
import subprocess

SOURCE_COMMIT = "5debc0e4ed3588748b8491c94c072df75a149834"
FIX_COMMIT = "ffe4a8d9e23f5bda924c1c17c83f5981adce7f89"
SOURCE_RUN = "34317875969"
PATH = "omnihub/src/main/java/org/eclipse/deeplearning4j/omnihub/ProgressInputStream.java"
BEFORE = b"    protected synchronized void afterRead(int n) {\n"
AFTER = b"    protected synchronized void afterRead(int n) throws IOException {\n"
LINE = 41


def repaired(original):
    lines = original.splitlines(keepends=True)
    if len(lines) < LINE or lines[LINE - 1] != BEFORE:
        raise ValueError("audited compiled-source declaration does not match")
    lines[LINE - 1] = AFTER
    return b"".join(lines)


def prepare(source, fix_source, fix_commit, commit, run_id, output):
    if (commit != SOURCE_COMMIT or fix_commit != FIX_COMMIT
            or not run_id.startswith(f"github-{SOURCE_RUN}-")
            or not run_id.removeprefix(f"github-{SOURCE_RUN}-").isdigit()
            or int(run_id.removeprefix(f"github-{SOURCE_RUN}-")) < 1):
        raise ValueError("compiled-source recovery requires audited source, worker run and fix SHA")
    for root, sha in ((source, commit), (fix_source, fix_commit)):
        if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() != sha:
            raise ValueError("compiled-source checkout revision mismatch")
    original = subprocess.check_output(["git", "show", f"{commit}:{PATH}"], cwd=source)
    fixed = subprocess.check_output(["git", "show", f"{fix_commit}:{PATH}"], cwd=fix_source)
    path = source / PATH
    parents = [source.joinpath(*path.relative_to(source).parts[:i])
               for i in range(len(path.relative_to(source).parts) + 1)]
    if any(parent.is_symlink() for parent in parents) or path.read_bytes() != original:
        raise ValueError("compiled-source recovery requires pristine source")
    if fixed != repaired(original):
        raise ValueError("source fix contains unaudited changes")
    provenance = {
        "schemaVersion": 1, "policy": "authorized-compiled-java-exception-declaration-v1",
        "sourceCommit": commit, "sourceRunId": SOURCE_RUN, "sourceFixCommit": fix_commit,
        "nativeWorkerReceiptsUnchanged": True, "compiledJavaChanged": True,
        "diagnostics": {"recoveryRunId": "34411560183", "jobId": "102667000677",
                        "error": "unreported exception java.io.IOException at super.afterRead(n)"},
        "files": [{"path": PATH, "line": LINE, "before": BEFORE.decode(), "after": AFTER.decode(),
                   "sourceSha256": hashlib.sha256(original).hexdigest(),
                   "repairedSha256": hashlib.sha256(fixed).hexdigest()}],
    }
    path.write_bytes(fixed)
    (output / "source-recovery-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance
