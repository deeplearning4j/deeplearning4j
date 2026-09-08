#!/usr/bin/env python3
"""Exercise production receipt comparison without launching SDK producers."""
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / "main/android/build-android-aot-sdk.sh"


def function_source(name):
    source = SCRIPT.read_text()
    start = source.index(name + "() {\n")
    end = source.index("\n}\n", start) + 3
    return source[start:end]


class CacheDiagnosticsTest(unittest.TestCase):
    def compare(self, receipt, expected):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "build-receipt"
            path.write_text(receipt)
            script = "set -euo pipefail\n" + function_source("receipt_has")
            script += "\n" + function_source("compatible_native_image_identity")
            # Identity is supplied as a positional argument, never shell source.
            script += '\nEXPECTED="$2"\nsdx_native_image_object_identity_lines() { printf "%s\\n" "$EXPECTED"; }\n'
            script += 'compatible_native_image_identity "$1"\n'
            return subprocess.run(["bash", "-c", script, "test", str(path), expected],
                                  capture_output=True, text=True, timeout=5)

    def test_source_only_change_remains_compatible(self):
        result = self.compare("source_manifest_sha256=old\nclasses_sha256=same\n",
                              "source_manifest_sha256=new\nclasses_sha256=same")
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual("", result.stderr)

    def test_all_changed_fields_reported_and_rejected(self):
        result = self.compare("classes_sha256=old\nmodel_classes_sha256=old-model\n",
                              "classes_sha256=new\nmodel_classes_sha256=new-model")
        self.assertEqual(1, result.returncode)
        self.assertIn("classes_sha256 cached=old current=new", result.stderr)
        self.assertIn("model_classes_sha256 cached=old-model current=new-model", result.stderr)

    def test_missing_field_is_not_accepted(self):
        result = self.compare("classes_sha256=same\n", "classes_sha256=same\ngraalvm_version_sha256=graal")
        self.assertEqual(1, result.returncode)
        self.assertIn("graalvm_version_sha256 cached=<missing> current=graal", result.stderr)


if __name__ == "__main__":
    unittest.main()
