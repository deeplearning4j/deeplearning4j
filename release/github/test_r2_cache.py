#!/usr/bin/env python3
"""R2 cache contracts. Run on GitHub runners from platform-tests, never locally."""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


worker = load("r2_prepare_worker", "release/github/prepare-worker.py")
builder = load("r2_build_platform", "release/aws/build-platform.py")
transport = load("r2_cloud_io", "release/aws/cloud-io.py")
dependency = load("r2_dependency_cache", "release/azure/dependency-cache.py")


def cache_config():
    return {
        "backend": "s3", "provider": "r2", "bucket": worker.R2_BUCKET,
        "endpoint": worker.R2_ENDPOINT, "region": "auto",
        "keyPrefix": worker.AZURE_CACHE_PREFIX, "serverSideEncryption": False,
        "accessKeyIdEnv": "R2_ACCESS_KEY_ID", "secretAccessKeyEnv": "R2_SECRET_ACCESS_KEY",
        "snapshotIdentityBackend": "azure",
        "toolchainCache": {"schemaVersion": 1, "keyPrefix": "deeplearning4j/releases/toolchain-cache/v1"},
    }


def cache_env():
    return {
        "R2_ACCESS_KEY_ID": " key\n", "R2_SECRET_ACCESS_KEY": " secret\r\n",
        "DL4J_CLOUD_IO": str(ROOT / "release/aws/cloud-io.py"),
        "DL4J_DEPENDENCY_CACHE_HELPER": str(ROOT / "release/azure/dependency-cache.py"),
    }


class R2CacheTests(unittest.TestCase):
    def test_explicit_credentials_are_sanitized_and_other_providers_removed(self):
        env = cache_env()
        env.update({"SCCACHE_GHA_ENABLED": "true", "SCCACHE_AZURE_CONNECTION_STRING": "old",
                    "AWS_SESSION_TOKEN": "old", "GITHUB_ACTIONS": "true"})
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            builder.configure_s3_cache_environment(cache_config(), env)
        self.assertEqual("key", env["AWS_ACCESS_KEY_ID"])
        self.assertEqual("secret", env["AWS_SECRET_ACCESS_KEY"])
        self.assertIn("::add-mask::secret", output.getvalue())
        self.assertNotIn("AWS_SESSION_TOKEN", env)
        self.assertNotIn("SCCACHE_GHA_ENABLED", env)
        self.assertNotIn("SCCACHE_AZURE_CONNECTION_STRING", env)
        self.assertEqual(worker.R2_ENDPOINT, env["SCCACHE_ENDPOINT"])
        self.assertEqual(worker.R2_ENDPOINT, env["DL4J_S3_ENDPOINT"])

    def test_credentials_fail_closed_even_with_other_provider_credentials(self):
        for value in ("", " \n", "abc\ndef"):
            env = cache_env()
            env.update({"R2_ACCESS_KEY_ID": value, "AWS_ACCESS_KEY_ID": "aws",
                        "SCCACHE_AZURE_CONNECTION_STRING": "azure"})
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                builder.configure_s3_cache_environment(cache_config(), env)
            env["DL4J_CACHE_BACKEND"] = "r2"
            with self.assertRaises(ValueError):
                transport.credentials(env)

    def test_sccache_uses_s3_and_no_aws_sse(self):
        env = cache_env()
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(builder, "ensure_cached_sccache", return_value="sccache"), \
                 patch.object(builder, "run"):
                builder.configure_compiler_cache({"compilerCache": cache_config()}, Path(temporary), env)
        self.assertEqual("disk,s3", env["SCCACHE_MULTILEVEL_CHAIN"])
        self.assertEqual("all", env["SCCACHE_MULTILEVEL_WRITE_ERROR_POLICY"])
        self.assertEqual("false", env["SCCACHE_S3_SERVER_SIDE_ENCRYPTION"])
        self.assertEqual(worker.AZURE_CACHE_PREFIX, env["SCCACHE_S3_KEY_PREFIX"])
        self.assertEqual("auto", env["SCCACHE_REGION"])

    def test_toolchain_transport_uses_same_keys_and_requires_adapter(self):
        env = cache_env()
        result = builder.toolchain_cache_transport({"compilerCache": cache_config()}, env)
        self.assertEqual("dl4j-cache", result[1])
        self.assertEqual("deeplearning4j/releases/toolchain-cache/v1", result[2])
        env.pop("DL4J_CLOUD_IO")
        with self.assertRaises(ValueError):
            builder.toolchain_cache_transport({"compilerCache": cache_config()}, env)

    def test_snapshot_identity_remains_compatible_with_migrated_azure_indexes(self):
        r2 = {"compilerCache": cache_config(), "shard": {"id": "sbsa", "contractDigest": "a" * 64}}
        r2["compilerCache"]["localSnapshot"] = {"schemaVersion": 1, "name": "ccache-l0", "refresh": True}
        azure = copy.deepcopy(r2)
        azure["compilerCache"]["backend"] = "azure"
        azure["compilerCache"].pop("snapshotIdentityBackend")
        self.assertEqual(builder.compiler_cache_snapshot_identity(azure),
                         builder.compiler_cache_snapshot_identity(r2))

    def test_private_manifest_strips_old_azure_base_url_preserving_keys(self):
        manifest = copy.deepcopy(worker.PUBLIC_DEPENDENCY_CACHE)
        manifest["schemaVersion"] = 1
        def download(command, **kwargs):
            self.assertEqual("r2", kwargs["env"]["DL4J_CACHE_BACKEND"])
            self.assertIn("dl4j-cache", command)
            Path(command[command.index("--file") + 1]).write_text(json.dumps(manifest))
        with patch.object(worker.subprocess, "run", side_effect=download):
            result = worker.load_r2_dependency_cache()
        self.assertNotIn("publicBaseUrl", result)
        self.assertEqual(manifest["host"], result["host"])
        self.assertEqual(manifest["targets"], result["targets"])
        with patch.object(worker.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "download")):
            with self.assertRaises(subprocess.CalledProcessError):
                worker.load_r2_dependency_cache()

    def test_s3_client_uses_sigv4_path_style_and_no_ambient_session_token(self):
        env = cache_env()
        builder.configure_s3_cache_environment(cache_config(), env)
        env["AWS_SESSION_TOKEN"] = "stale"
        with patch("boto3.client") as create:
            transport.client(env)
        kwargs = create.call_args.kwargs
        self.assertEqual(worker.R2_ENDPOINT, kwargs["endpoint_url"])
        self.assertEqual("key", kwargs["aws_access_key_id"])
        self.assertIsNone(kwargs["aws_session_token"])
        self.assertEqual("s3v4", kwargs["config"].signature_version)
        self.assertEqual("path", kwargs["config"].s3["addressing_style"])

    def test_only_object_not_found_is_a_miss(self):
        from botocore.exceptions import ClientError
        for code, missing in (("404", True), ("403", False), ("SignatureDoesNotMatch", False)):
            fake = MagicMock()
            fake.download_file.side_effect = ClientError({"Error": {"Code": code}}, "GetObject")
            with tempfile.TemporaryDirectory() as temporary, patch.object(transport, "client", return_value=fake):
                path = Path(temporary) / "download"
                if missing:
                    self.assertEqual(3, transport.transfer("download", "bucket", "key", path, 3))
                else:
                    with self.assertRaises(RuntimeError):
                        transport.transfer("download", "bucket", "key", path, 3)

    def test_upload_uses_managed_multipart_without_sse(self):
        fake = MagicMock()
        with patch.object(transport, "client", return_value=fake):
            transport.transfer("upload", "bucket", "key", Path("archive"))
        args = fake.upload_file.call_args
        self.assertEqual(("archive", "bucket", "key"), args.args)
        self.assertNotIn("ExtraArgs", args.kwargs)
        self.assertEqual(64 * 1024 * 1024, args.kwargs["Config"].multipart_chunksize)

    def test_migration_dispatch_is_exclusive_with_build_and_publication(self):
        workflow = (ROOT / ".github/workflows/build-deploy-cross-platform.yml").read_text()
        for name in ("java-only", "release", "publish-staged"):
            block = workflow.split(f"  {name}:\n", 1)[1].split("    uses:", 1)[0]
            self.assertIn("(inputs.cacheMigration == '' || inputs.cacheMigration == 'none')", block)
        migration = workflow.split("  cache-migration:\n", 1)[1].split("    uses:", 1)[0]
        self.assertIn("inputs.cacheMigration != '' && inputs.cacheMigration != 'none'", migration)

    def test_current_actions_select_r2_without_starting_gha(self):
        action = (ROOT / ".github/actions/run-release-worker/action.yml").read_text()
        self.assertIn("args+=(--r2-cache)", action)
        self.assertNotIn("args+=(--azure-cache)", action)
        self.assertEqual(3, action.count("start-server: 'false'"))
        workflow = (ROOT / ".github/workflows/_release-worker.yml").read_text()
        self.assertEqual(2, workflow.count("r2-access-key-id: ${{ secrets.R2_ACCESS_KEY_ID }}"))
        self.assertNotIn("azure-cache-connection-string:", workflow)


@unittest.skipUnless(os.environ.get("DL4J_R2_LIVE_VALIDATION") == "1", "remote authenticated validation only")
class R2LiveCacheTests(unittest.TestCase):
    def live_env(self):
        env = os.environ.copy()
        env.update({"DL4J_CACHE_BACKEND": "r2", "DL4J_S3_ENDPOINT": worker.R2_ENDPOINT,
                    "DL4J_S3_REGION": "auto"})
        return env

    def test_migrated_namespaces(self):
        storage = transport.client(self.live_env())
        for namespace in ("compiler-cache/v1", "toolchain-cache/v1", "dependency-cache/v2"):
            with self.subTest(namespace=namespace):
                prefix = "deeplearning4j/releases/" + namespace + "/"
                pages = storage.get_paginator("list_objects_v2").paginate(
                    Bucket=worker.R2_BUCKET, Prefix=prefix, PaginationConfig={"PageSize": 10}
                )
                key = next((item["Key"] for page in pages for item in page.get("Contents", [])
                            if "/r2-transport-contract/" not in item["Key"]), None)
                self.assertIsNotNone(key, f"No migrated objects in {prefix}")
                storage.head_object(Bucket=worker.R2_BUCKET, Key=key)
                print(f"R2 namespace readable: {prefix}")

    def test_migrated_manifest_references(self):
        env = self.live_env()
        storage = transport.client(env)
        with patch.dict(os.environ, env):
            manifest = worker.load_r2_dependency_cache()
        self.assertNotIn("publicBaseUrl", manifest)
        for item in [manifest["host"], *manifest["targets"]]:
            for field in ("indexObject", "archiveObject"):
                storage.head_object(Bucket=worker.R2_BUCKET, Key=item[field])

    def test_archive_roundtrip(self):
        env = self.live_env()
        # Exercise actual publish/restore (including digest/member verification).
        # Add only content-addressed test objects; never delete any remote key.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "payload.txt").write_text("R2 worker archive contract v1\n")
            identity = dependency.canonical_identity({"contract": "r2-worker-transport-v1"})
            common = [sys.executable, str(ROOT / "release/azure/dependency-cache.py")]
            options = ["--cloud-io", str(ROOT / "release/aws/cloud-io.py"),
                       "--bucket", worker.R2_BUCKET,
                       "--prefix", "deeplearning4j/releases/toolchain-cache/v1",
                       "--name", "r2-transport-contract", "--identity", identity]
            subprocess.run(common + ["publish"] + options + ["--source", str(source)], env=env, check=True)
            destination = root / "restored"
            subprocess.run(common + ["restore"] + options + ["--destination", str(destination)], env=env, check=True)
            self.assertEqual((source / "payload.txt").read_bytes(), (destination / "payload.txt").read_bytes())
        print("R2 authenticated verified archive roundtrip passed")


if __name__ == "__main__":
    unittest.main()
