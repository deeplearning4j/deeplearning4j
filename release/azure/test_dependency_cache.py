#!/usr/bin/env python3
"""Tests for independently addressable release-worker dependency archives."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cache = load_module("dl4j_azure_dependency_cache", HERE / "dependency-cache.py")


FAKE_CLOUD_IO = """#!/usr/bin/env python3
import argparse
from pathlib import Path
import shutil

root = argparse.ArgumentParser()
commands = root.add_subparsers(dest="command", required=True)
for name in ("upload", "download"):
    command = commands.add_parser(name)
    command.add_argument("--bucket", required=True)
    command.add_argument("--object", required=True)
    command.add_argument("--file", required=True)
    command.add_argument("--client-id")
    command.add_argument("--missing-exit-code", type=int, default=1)
args = root.parse_args()
bucket = Path(args.bucket)
target = bucket / args.object
if args.command == "upload":
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.file, target)
    with (bucket / "operations.log").open("a", encoding="utf-8") as stream:
        stream.write("upload " + args.object + "\\n")
else:
    if (bucket / "force-error").exists():
        raise SystemExit(17)
    if not target.is_file():
        raise SystemExit(args.missing_exit_code)
    Path(args.file).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target, args.file)
"""


class DependencyCacheIntegrationContractTests(unittest.TestCase):
    def test_worker_caches_each_immutable_bootstrap_dependency(self):
        worker = (HERE / "worker.sh").read_text(encoding="utf-8")
        for name in (
            "protobuf",
            "protoc",
            "cmake",
            "rust-cbindgen",
            "android-ndk",
            "container-image",
        ):
            self.assertIn(name, worker)
        self.assertIn("__DL4J_DEPENDENCY_CACHE_B64__", worker)
        self.assertIn("DL4J_DEPENDENCY_CACHE_HELPER=/dl4j-dependency-cache.py", worker)
        self.assertIn("ensure_container_image", worker)
        self.assertIn("--default-toolchain 1.97.1", worker)
        self.assertIn("--version 0.29.4 cbindgen", worker)

        windows_worker = (HERE / "worker.ps1").read_text(encoding="utf-8")
        self.assertIn("__DL4J_DEPENDENCY_CACHE_B64__", windows_worker)
        self.assertIn("DL4J_DEPENDENCY_CACHE_HELPER", windows_worker)

    def test_rocm_sdk_uses_the_same_content_addressed_cache_contract(self):
        driver = (HERE.parent / "aws" / "build-platform.py").read_text(encoding="utf-8")
        self.assertIn('toolchain_cache_identity(\n        "rocm-sdk"', driver)
        self.assertIn('name="rocm-sdk"', driver)
        self.assertIn("restore_toolchain_dependency(", driver)
        self.assertIn("publish_toolchain_dependency(", driver)
        self.assertIn('"destination": "/opt/rocm"', driver)
        self.assertIn('name="sccache"', driver)
        self.assertIn("ensure_cached_sccache(cache_dir, config, env)", driver)
        self.assertIn('name="sccache-l0"', driver)
        self.assertIn("restore_compiler_cache_snapshot(config, env, cache_dir)", driver)
        self.assertIn("publish_compiler_cache_snapshot(", driver)
        self.assertIn('name="openblas"', driver)

    def test_controller_embeds_helper_and_advertises_versioned_prefix(self):
        source = (HERE / "release.py").read_text(encoding="utf-8")
        self.assertIn("__DL4J_DEPENDENCY_CACHE_B64__", source)
        self.assertIn('"toolchainCache": {', source)
        self.assertIn('"localSnapshot": {', source)
        self.assertIn('"/toolchain-cache/v1"', source)


class DependencyCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.cloud = self.root / "cloud-io.py"
        self.cloud.write_text(FAKE_CLOUD_IO, encoding="utf-8")
        self.cloud.chmod(0o755)
        self.bucket = self.root / "bucket"
        self.bucket.mkdir()
        self.identity = cache.canonical_identity(
            {"name": "protobuf", "version": "3.8.0", "platform": "linux-x86_64"}
        )

    def tearDown(self):
        self.temporary.cleanup()

    def arguments(self, command: str, **values):
        common = {
            "cloud_io": self.cloud,
            "bucket": str(self.bucket),
            "prefix": "release-cache/v1",
            "name": "protobuf",
            "identity": self.identity,
            "client_id": "client",
        }
        common.update(values)
        return type("Arguments", (), common)()

    def test_publish_then_restore_preserves_files_and_safe_symlinks(self):
        source = self.root / "source"
        (source / "bin").mkdir(parents=True)
        executable = source / "bin" / "protoc-3.8.0"
        executable.write_text("binary", encoding="utf-8")
        executable.chmod(0o755)
        os.symlink("protoc-3.8.0", source / "bin" / "protoc")

        self.assertEqual(0, cache.publish(self.arguments("publish", source=source)))
        operations = (self.bucket / "operations.log").read_text(encoding="utf-8").splitlines()
        self.assertIn("/archives/protobuf/sha256/", operations[0])
        self.assertTrue(operations[-1].endswith(f"/indexes/protobuf/{self.identity}.json"))

        destination = self.root / "restored"
        self.assertEqual(
            0,
            cache.restore(self.arguments("restore", destination=destination)),
        )
        self.assertEqual("binary", (destination / "bin" / "protoc").read_text())
        self.assertTrue((destination / "bin" / "protoc").is_symlink())
        marker = json.loads(
            (destination / ".dl4j-dependency-cache.json").read_text(encoding="utf-8")
        )
        self.assertEqual(self.identity, marker["identity"])

        operation_count = len(
            (self.bucket / "operations.log").read_text(encoding="utf-8").splitlines()
        )
        self.assertEqual(
            0,
            cache.restore(self.arguments("restore", destination=destination)),
        )
        self.assertEqual(
            operation_count,
            len((self.bucket / "operations.log").read_text(encoding="utf-8").splitlines()),
        )

    def test_publish_normalizes_absolute_symlink_within_dependency(self):
        source = self.root / "source"
        (source / "lib").mkdir(parents=True)
        library = source / "lib" / "libdependency.so.1"
        library.write_text("binary", encoding="utf-8")
        os.symlink(str(library), source / "lib" / "libdependency.so")

        self.assertEqual(0, cache.publish(self.arguments("publish", source=source)))
        destination = self.root / "restored"
        self.assertEqual(0, cache.restore(self.arguments("restore", destination=destination)))
        alias = destination / "lib" / "libdependency.so"
        self.assertTrue(alias.is_symlink())
        self.assertEqual("binary", alias.read_text(encoding="utf-8"))

    def test_transport_failure_is_not_treated_as_cache_miss(self):
        (self.bucket / "force-error").touch()
        destination = self.root / "restored"
        with self.assertRaisesRegex(Exception, "exit status 17"):
            cache.restore(self.arguments("restore", destination=destination))
        self.assertFalse(destination.exists())

    def test_missing_index_is_an_explicit_cache_miss(self):
        destination = self.root / "restored"
        self.assertEqual(
            cache.MISS_EXIT_CODE,
            cache.restore(self.arguments("restore", destination=destination)),
        )
        self.assertFalse(destination.exists())

    def test_corrupt_archive_is_rejected_without_exposing_destination(self):
        source = self.root / "source"
        source.mkdir()
        (source / "value").write_text("valid", encoding="utf-8")
        cache.publish(self.arguments("publish", source=source))
        index_path = (
            self.bucket
            / "release-cache/v1"
            / "indexes"
            / "protobuf"
            / f"{self.identity}.json"
        )
        index = json.loads(index_path.read_text(encoding="utf-8"))
        archive = self.bucket / index["archiveObject"]
        archive.write_bytes(b"corrupt")
        destination = self.root / "restored"
        with self.assertRaisesRegex(RuntimeError, "attestation mismatch"):
            cache.restore(self.arguments("restore", destination=destination))
        self.assertFalse(destination.exists())

    def test_identity_is_canonical_and_order_independent(self):
        left = cache.canonical_identity({"version": "1", "components": ["a", "b"]})
        right = cache.canonical_identity({"components": ["a", "b"], "version": "1"})
        self.assertEqual(left, right)
        self.assertNotEqual(
            left,
            cache.canonical_identity({"version": "1", "components": ["b", "a"]}),
        )


if __name__ == "__main__":
    unittest.main()
