"""Snapshot publication regressions; all network calls are mocked."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from release.central import verify_java_snapshots as publication


class JavaSnapshotPublicationTests(unittest.TestCase):
    def log(self, artifacts=publication.REQUIRED_PARENTS):
        return "\n".join(
            f"[INFO] Installing /source/pom.xml to /home/runner/.m2/repository/"
            f"org/eclipse/deeplearning4j/{artifact}/1.0.0-SNAPSHOT/{artifact}-1.0.0-SNAPSHOT.pom"
            for artifact in artifacts)

    def test_inventory_includes_all_parent_poms_not_only_staged_artifacts(self):
        poms = publication.inventory(self.log(), "1.0.0-SNAPSHOT")
        self.assertEqual(3, len(poms))
        self.assertTrue(any("/nd4j-api-parent/" in path for path in poms))

    def test_missing_or_skipped_parents_fail(self):
        with self.assertRaisesRegex(ValueError, "Missing required reactor parent"):
            publication.inventory(self.log({"nd4j-api"}), "1.0.0-SNAPSHOT")
        with self.assertRaisesRegex(ValueError, "skipped snapshot publication"):
            publication.inventory(self.log() + "\nSkipping Central Snapshot Publishing", "1.0.0-SNAPSHOT")
        with self.assertRaises(ValueError):
            publication.inventory(self.log(), "2.0.0-SNAPSHOT")

    def test_remote_pom_must_match_this_build(self):
        metadata = b'''<metadata><versioning><snapshotVersions><snapshotVersion>
          <extension>pom</extension><value>1.0.0-20260912.030338-1</value>
          </snapshotVersion></snapshotVersions></versioning></metadata>'''
        with tempfile.TemporaryDirectory() as directory:
            pom = Path(directory) / "parent.pom"
            pom.write_bytes(b"<project/>")
            poms = {"org/eclipse/deeplearning4j/nd4j-api-parent/1.0.0-SNAPSHOT/nd4j-api-parent-1.0.0-SNAPSHOT.pom": pom}
            with patch.object(publication, "fetch", side_effect=[metadata, pom.read_bytes()]) as fetch:
                publication.verify(poms, attempts=1)
                self.assertTrue(fetch.call_args.args[0].endswith("nd4j-api-parent-1.0.0-20260912.030338-1.pom"))
            with patch.object(publication, "fetch", side_effect=[metadata, b"stale"]):
                with self.assertRaisesRegex(ValueError, "differs from this build"):
                    publication.verify(poms, attempts=1)
            with patch.object(publication, "fetch", return_value=b"<metadata/>"):
                with self.assertRaisesRegex(ValueError, "Missing or ambiguous"):
                    publication.verify(poms, attempts=1)
            with patch.object(publication, "fetch", side_effect=OSError("404")):
                with self.assertRaises(OSError):
                    publication.verify(poms, attempts=1)
            with patch.object(publication, "fetch", side_effect=[OSError("404"), metadata, pom.read_bytes()]), \
                    patch.object(publication.time, "sleep") as sleep:
                publication.verify(poms, attempts=2)
                sleep.assert_called_once()


if __name__ == "__main__":
    unittest.main()
