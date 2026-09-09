"""Audited Javadoc-only repair overlay for the pinned native release source.

Only the seven reviewed Javadoc lines are eligible. No whole fix checkout is
merged: unrelated changes at that revision cannot enter the Java reactor.
Extending this table requires reviewing the original comment and source SHA.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess

SOURCE_COMMIT = "5debc0e4ed3588748b8491c94c072df75a149834"
API = "nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/"
REPAIRS = {
    API + "autodiff/samediff/SameDiff.java": {2292: "Example Usage:"},
    API + "linalg/api/ndarray/SparseNDArray.java": {
        115: "CSR ({@link SparseFormat#CSR})", 121: "CSC ({@link SparseFormat#CSC})", 209: "BSR layout"},
    API + "linalg/api/ndarray/SparseSolvers.java": {212: "Algorithm", 385: "Steps"},
}
# Each entry is an exact original/replacement pair; preserve every other byte.
REPAIRS = {name: {number: ("     * <h3>" + heading + "</h3>\n",
                          "     * <h4>" + heading + "</h4>\n")
                  for number, heading in repairs.items()}
           for name, repairs in REPAIRS.items()}
REPAIRS["nd4j/nd4j-tensorflow/src/main/java/org/nd4j/tensorflow/conversion/TensorflowConversion.java"] = {
    356: ("     * @throws IOException\n",
          "     * @throws IllegalStateException if TensorFlow cannot import the graph\n"),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def repaired(original, repairs):
    lines = original.splitlines(keepends=True)
    for number, (before, after) in repairs.items():
        expected = before.encode()
        if number > len(lines) or lines[number - 1] != expected:
            raise ValueError(f"audited Javadoc line {number} does not match")
        lines[number - 1] = after.encode()
    return b"".join(lines)


def prepare(source, fix_source, fix_commit, commit, output):
    if commit != SOURCE_COMMIT or not re.fullmatch(r"[0-9a-f]{40}", fix_commit):
        raise ValueError("documentation repair requires the audited source and immutable fix SHA")
    for root, sha in ((source, commit), (fix_source, fix_commit)):
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        if actual != sha:
            raise ValueError("documentation checkout revision mismatch")
    pending, evidence = [], []
    for name, repairs in REPAIRS.items():
        original = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=source)
        fixed = subprocess.check_output(["git", "show", f"{fix_commit}:{name}"], cwd=fix_source)
        path = source / name
        if path.is_symlink() or path.read_bytes() != original:
            raise ValueError(f"documentation source is not pristine: {name}")
        expected = repaired(original, repairs)
        if fixed != expected:
            raise ValueError(f"fix contains unaudited or compiled-code changes: {name}")
        pending.append((path, fixed))
        evidence.append({"path": name, "sourceSha256": digest(original), "repairedSha256": digest(fixed),
                         "lines": sorted(repairs), "changes": [
                             {"line": number, "before": before, "after": after}
                             for number, (before, after) in sorted(repairs.items())],
                         "change": "Audited Javadoc lines only; all other bytes unchanged"})
    provenance = {"schemaVersion": 1, "sourceCommit": commit, "documentationFixCommit": fix_commit,
                  "policy": "audited-javadoc-only-v2", "files": evidence}
    # Validate every file before writing any overlay. Preserve line numbers and
    # every non-repaired byte, including all compiled code and source positions.
    for path, fixed in pending:
        path.write_bytes(fixed)
    (output / "documentation-recovery-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance
