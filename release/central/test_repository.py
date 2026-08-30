#!/usr/bin/env python3
"""Tests for shared Maven repository materialization."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET
import zipfile

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

            root_index = output / "index.html"
            coordinate_index = (
                output / "org/eclipse/deeplearning4j/nd4j-cuda-12.9/index.html"
            )
            self.assertTrue(root_index.is_file())
            self.assertTrue(coordinate_index.is_file())
            root_html = root_index.read_text(encoding="utf-8")
            coordinate_html = coordinate_index.read_text(encoding="utf-8")
            self.assertIn('href="org/"', root_html)
            self.assertIn('href="1.0.0-SNAPSHOT/"', coordinate_html)
            self.assertNotIn(r"\n", root_html)
            self.assertIn("\n", root_html)

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

    def test_materializer_restores_the_component_pom_embedded_in_a_jar(self):
        version = "1.0.0"
        artifact_id = "libtokenizers"
        embedded_pom = (
            "<project xmlns=\"http://maven.apache.org/POM/4.0.0\">"
            "<modelVersion>4.0.0</modelVersion>"
            f"<groupId>org.eclipse.deeplearning4j</groupId>"
            f"<artifactId>{artifact_id}</artifactId>"
            f"<version>{version}</version>"
            "</project>\n"
        ).encode("utf-8")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = (
                root
                / "shard/org/eclipse/deeplearning4j"
                / artifact_id
                / version
            )
            source.mkdir(parents=True)
            jar = source / f"{artifact_id}-{version}-linux-arm64.jar"
            with zipfile.ZipFile(jar, "w") as archive:
                archive.writestr(
                    "META-INF/maven/org.eclipse.deeplearning4j/"
                    f"{artifact_id}/pom.xml",
                    embedded_pom,
                )

            output = root / "repository"
            manifest_path = root / "repository-manifest.json"
            manifest = repository.materialize_test_repository(
                [root / "shard"],
                output,
                manifest_path,
                version,
                "feedface",
                metadata_updated="20260811000000",
            )

            restored = (
                output
                / "org/eclipse/deeplearning4j"
                / artifact_id
                / version
                / f"{artifact_id}-{version}.pom"
            )
            self.assertEqual(embedded_pom, restored.read_bytes())
            self.assertTrue(Path(str(restored) + ".sha256").is_file())
            self.assertIn(
                restored.relative_to(output).as_posix(),
                {item["path"] for item in manifest["files"]},
            )
            repository.verify(output, manifest_path, version, "feedface")

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


class MavenExecutableTests(unittest.TestCase):
    def test_windows_maven_cmd_uses_command_interpreter(self):
        with (
            mock.patch.object(
                repository.shutil,
                "which",
                return_value=r"C:\hostedtoolcache\windows\maven\bin\mvn.cmd",
            ),
            mock.patch.dict(
                repository.os.environ,
                {"COMSPEC": r"C:\Windows\System32\cmd.exe"},
            ),
        ):
            self.assertEqual(
                [
                    r"C:\Windows\System32\cmd.exe",
                    "/d",
                    "/c",
                    r"C:\hostedtoolcache\windows\maven\bin\mvn.cmd",
                ],
                repository.resolve_maven_command("mvn", os_name="nt"),
            )

    def test_native_maven_executable_is_used_directly(self):
        with mock.patch.object(
            repository.shutil,
            "which",
            return_value="/opt/apache-maven/bin/mvn",
        ):
            self.assertEqual(
                ["/opt/apache-maven/bin/mvn"],
                repository.resolve_maven_command("mvn", os_name="posix"),
            )

    def test_missing_maven_executable_has_actionable_error(self):
        with mock.patch.object(repository.shutil, "which", return_value=None):
            with self.assertRaisesRegex(FileNotFoundError, "was not found on PATH"):
                repository.resolve_maven_command("mvn", os_name="nt")

    def test_snapshot_deploy_uses_resolved_windows_command_prefix(self):
        version = "1.0.0-SNAPSHOT"
        with tempfile.TemporaryDirectory() as temporary:
            version_dir = (
                Path(temporary)
                / "org/eclipse/deeplearning4j/nd4j-native"
                / version
            )
            version_dir.mkdir(parents=True)
            base = version_dir / f"nd4j-native-{version}"
            Path(str(base) + ".pom").write_text(
                "<project><modelVersion>4.0.0</modelVersion></project>\n",
                encoding="utf-8",
            )
            Path(str(base) + "-windows-x86_64-avx512.jar").write_bytes(
                b"PK\x03\x04test-jar"
            )
            prefix = ["cmd.exe", "/d", "/c", "mvn.cmd"]
            with (
                mock.patch.object(
                    repository,
                    "resolve_maven_command",
                    return_value=prefix,
                ),
                mock.patch.object(repository.subprocess, "run") as run,
            ):
                repository.deploy_snapshot(
                    Path(temporary),
                    version,
                    "central-portal-snapshots",
                    "https://example.invalid/snapshots/",
                )

            command = run.call_args.args[0]
            self.assertEqual(prefix, command[:len(prefix)])
            self.assertIn(
                "-Dclassifiers=windows-x86_64-avx512",
                command,
            )
            run.assert_called_once_with(command, check=True)

    def test_merged_shards_deploy_all_classifiers_for_one_gav(self):
        version = "1.0.0-SNAPSHOT"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "worker-linux-rocm-6"
            second = root / "worker-linux-rocm-7"
            third = root / "worker-linux-rocm-10"
            relative = Path(
                "org/eclipse/deeplearning4j/nd4j-zluda-12.9"
            ) / version
            for source in (first, second, third):
                version_dir = source / relative
                version_dir.mkdir(parents=True)
                base = version_dir / f"nd4j-zluda-12.9-{version}"
                Path(str(base) + ".pom").write_text(
                    "<project><modelVersion>4.0.0</modelVersion></project>\n",
                    encoding="utf-8",
                )
            base_dir = first / relative
            base_name = "nd4j-zluda-12.9-" + version
            Path(str(base_dir / base_name) + ".jar").write_bytes(b"base")
            Path(str(base_dir / base_name) + "-linux-x86_64-zluda-rocm-6.2.4.jar").write_bytes(
                b"rocm6"
            )
            second_dir = second / relative
            Path(str(second_dir / base_name) + ".jar").write_bytes(b"base-rocm7")
            Path(
                str(second_dir / base_name)
                + "-linux-x86_64-zluda-rocm-7.2.4.jar"
            ).write_bytes(b"rocm7")
            third_dir = third / relative
            Path(str(third_dir / base_name) + ".jar").write_bytes(b"base-rocm10")
            Path(
                str(third_dir / base_name)
                + "-linux-x86_64-zluda-rocm-10.0.0.jar"
            ).write_bytes(b"rocm10")

            merged = root / "merged"
            manifest = root / "merged-manifest.json"
            repository.merge(
                [first, second, third],
                merged,
                manifest,
                version,
                "deadbeef",
                allow_unclassified_duplicates=True,
            )
            repository.verify(merged, manifest, version, "deadbeef")

            with (
                mock.patch.object(
                    repository,
                    "resolve_maven_command",
                    return_value=["mvn"],
                ),
                mock.patch.object(repository.subprocess, "run") as run,
            ):
                repository.deploy_snapshot(
                    merged,
                    version,
                    "central-portal-snapshots",
                    "https://example.invalid/snapshots/",
                )

            run.assert_called_once()
            command = run.call_args.args[0]
            self.assertIn(
                "-Dclassifiers=linux-x86_64-zluda-rocm-10.0.0,"
                "linux-x86_64-zluda-rocm-6.2.4,"
                "linux-x86_64-zluda-rocm-7.2.4",
                command,
            )


if __name__ == "__main__":
    unittest.main()
