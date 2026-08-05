#!/usr/bin/env python3
"""Tests for shared Maven repository materialization."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


repository = load_module("dl4j_central_repository", HERE / "repository.py")
NS = {"m": repository.MAVEN_METADATA_NAMESPACE}


class MavenMetadataTests(unittest.TestCase):
    def write_component(
        self,
        root: Path,
        *,
        group_path: str,
        artifact_id: str,
        version: str,
        classifier: str | None,
    ) -> Path:
        version_dir = root / group_path / artifact_id / version
        version_dir.mkdir(parents=True, exist_ok=True)
        base_name = f"{artifact_id}-{version}"
        (version_dir / f"{base_name}.pom").write_text(
            "<project><modelVersion>4.0.0</modelVersion></project>\n",
            encoding="utf-8",
        )
        jar_name = base_name + (f"-{classifier}" if classifier else "") + ".jar"
        (version_dir / jar_name).write_bytes(b"PK\x03\x04test-jar")
        return version_dir

    def test_snapshot_repository_has_a_and_v_metadata_with_stable_names(self):
        version = "1.0.0-SNAPSHOT"
        updated = "20260805010203"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "shard"
            version_dir = self.write_component(
                source,
                group_path="org/eclipse/deeplearning4j",
                artifact_id="nd4j-cuda-12.9",
                version=version,
                classifier="windows-x86_64-zluda",
            )
            output = root / "repository"
            manifest_path = root / "repository-manifest.json"

            manifest = repository.materialize_test_repository(
                [source],
                output,
                manifest_path,
                version,
                "deadbeef",
                metadata_updated=updated,
            )

            artifact_dir = output / version_dir.relative_to(source).parent
            published_version_dir = artifact_dir / version
            artifact_metadata_path = artifact_dir / "maven-metadata.xml"
            version_metadata_path = published_version_dir / "maven-metadata.xml"
            self.assertTrue(artifact_metadata_path.is_file())
            self.assertTrue(version_metadata_path.is_file())

            artifact_metadata = ET.parse(artifact_metadata_path).getroot()
            self.assertEqual("1.1.0", artifact_metadata.attrib["modelVersion"])
            self.assertEqual(
                "org.eclipse.deeplearning4j",
                artifact_metadata.findtext("m:groupId", namespaces=NS),
            )
            self.assertEqual(
                "nd4j-cuda-12.9",
                artifact_metadata.findtext("m:artifactId", namespaces=NS),
            )
            self.assertEqual(
                version,
                artifact_metadata.findtext("m:versioning/m:latest", namespaces=NS),
            )
            self.assertIsNone(
                artifact_metadata.find("m:versioning/m:release", namespaces=NS)
            )
            self.assertEqual(
                [version],
                [
                    node.text
                    for node in artifact_metadata.findall(
                        "m:versioning/m:versions/m:version", namespaces=NS
                    )
                ],
            )
            self.assertEqual(
                updated,
                artifact_metadata.findtext(
                    "m:versioning/m:lastUpdated", namespaces=NS
                ),
            )

            version_metadata = ET.parse(version_metadata_path).getroot()
            self.assertEqual(
                version, version_metadata.findtext("m:version", namespaces=NS)
            )
            self.assertEqual(
                "true",
                version_metadata.findtext(
                    "m:versioning/m:snapshot/m:localCopy", namespaces=NS
                ),
            )
            snapshot_versions = {
                (
                    node.findtext("m:extension", namespaces=NS),
                    node.findtext("m:classifier", namespaces=NS),
                    node.findtext("m:value", namespaces=NS),
                    node.findtext("m:updated", namespaces=NS),
                )
                for node in version_metadata.findall(
                    "m:versioning/m:snapshotVersions/m:snapshotVersion",
                    namespaces=NS,
                )
            }
            self.assertEqual(
                {
                    ("pom", None, version, updated),
                    ("jar", "windows-x86_64-zluda", version, updated),
                },
                snapshot_versions,
            )

            for metadata_path in (artifact_metadata_path, version_metadata_path):
                for algorithm in repository.CHECKSUMS:
                    checksum_path = Path(str(metadata_path) + f".{algorithm}")
                    self.assertEqual(
                        repository.digest(metadata_path, algorithm) + "\n",
                        checksum_path.read_text(encoding="ascii"),
                    )

            manifest_files = {item["path"] for item in manifest["files"]}
            self.assertIn(
                "org/eclipse/deeplearning4j/nd4j-cuda-12.9/maven-metadata.xml",
                manifest_files,
            )
            self.assertIn(
                "org/eclipse/deeplearning4j/nd4j-cuda-12.9/"
                f"{version}/maven-metadata.xml.sha512",
                manifest_files,
            )
            repository.verify(output, manifest_path, version, "deadbeef")
            manifest["files"][0]["size"] += 1
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "manifest mismatch"):
                repository.verify(output, manifest_path, version, "deadbeef")

    def test_release_repository_has_a_metadata_and_no_v_metadata(self):
        version = "1.2.3"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "shard"
            version_dir = self.write_component(
                source,
                group_path="org/nd4j",
                artifact_id="nd4j-native",
                version=version,
                classifier=None,
            )
            output = root / "repository"
            manifest_path = root / "repository-manifest.json"

            repository.materialize_test_repository(
                [source],
                output,
                manifest_path,
                version,
                "cafebabe",
                metadata_updated="20260805030405",
            )

            artifact_dir = output / version_dir.relative_to(source).parent
            artifact_metadata = ET.parse(
                artifact_dir / "maven-metadata.xml"
            ).getroot()
            self.assertEqual(
                version,
                artifact_metadata.findtext("m:versioning/m:latest", namespaces=NS),
            )
            self.assertEqual(
                version,
                artifact_metadata.findtext("m:versioning/m:release", namespaces=NS),
            )
            self.assertFalse((artifact_dir / version / "maven-metadata.xml").exists())
            repository.verify(output, manifest_path, version, "cafebabe")

    def test_metadata_timestamp_must_be_maven_utc_format(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_component(
                root,
                group_path="org/nd4j",
                artifact_id="nd4j-native",
                version="1.0.0-SNAPSHOT",
                classifier=None,
            )
            for invalid_timestamp in ("2026-08-05", "20261399000000"):
                with self.subTest(updated=invalid_timestamp):
                    with self.assertRaisesRegex(ValueError, "yyyyMMddHHmmss"):
                        repository.write_maven_metadata(
                            root,
                            "1.0.0-SNAPSHOT",
                            updated=invalid_timestamp,
                        )


if __name__ == "__main__":
    unittest.main()
