#!/usr/bin/env python3
"""Unit tests for the release provisioner's fail-closed AWS validation."""

import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
import urllib.error
import xml.etree.ElementTree as ET
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO, StringIO
from pathlib import Path
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("dl4j_aws_release", Path(__file__).with_name("release.py"))
release = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(release)
FORWARDER_SPEC = importlib.util.spec_from_file_location(
    "dl4j_aws_log_forwarder", Path(__file__).with_name("log-forwarder.py")
)
log_forwarder = importlib.util.module_from_spec(FORWARDER_SPEC)
assert FORWARDER_SPEC.loader is not None
FORWARDER_SPEC.loader.exec_module(log_forwarder)
BUILD_PLATFORM_SPEC = importlib.util.spec_from_file_location(
    "dl4j_aws_build_platform", Path(__file__).with_name("build-platform.py")
)
build_platform = importlib.util.module_from_spec(BUILD_PLATFORM_SPEC)
assert BUILD_PLATFORM_SPEC.loader is not None
BUILD_PLATFORM_SPEC.loader.exec_module(build_platform)


class FakeSsm:
    def get_parameter(self, **_kwargs):
        raise AssertionError("query-backed AMIs must not use an unverified SSM parameter")


class FakeEc2:
    def __init__(self, image=None, architectures=None, offered=True):
        self.image = image or {
            "ImageId": "ami-verified", "Name": "ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-20260701",
            "OwnerId": "099720109477", "Architecture": "x86_64", "State": "available",
            "RootDeviceType": "ebs", "VirtualizationType": "hvm", "CreationDate": "2026-07-01T00:00:00Z",
        }
        self.architectures = architectures or ["x86_64"]
        self.offered = offered

    def describe_images(self, **_kwargs):
        return {"Images": [self.image]}

    def describe_instance_types(self, InstanceTypes):
        return {"InstanceTypes": [{
            "InstanceType": item, "ProcessorInfo": {"SupportedArchitectures": self.architectures},
            "VCpuInfo": {"DefaultVCpus": 96},
        } for item in InstanceTypes]}

    def describe_instance_type_offerings(self, **kwargs):
        values = kwargs["Filters"][0]["Values"] if self.offered else []
        zone = kwargs["Filters"][1]["Values"][0]
        return {"InstanceTypeOfferings": [{"InstanceType": item, "Location": zone} for item in values]}


class SizingEc2:
    def describe_instance_types(self, **_kwargs):
        return {"InstanceTypes": [
            {"InstanceType": "c7i.large", "ProcessorInfo": {"SupportedArchitectures": ["x86_64"]},
             "VCpuInfo": {"DefaultVCpus": 2}, "MemoryInfo": {"SizeInMiB": 4096}},
            {"InstanceType": "c7i.xlarge", "ProcessorInfo": {"SupportedArchitectures": ["x86_64"]},
             "VCpuInfo": {"DefaultVCpus": 4}, "MemoryInfo": {"SizeInMiB": 8192}},
            {"InstanceType": "c7i.2xlarge", "ProcessorInfo": {"SupportedArchitectures": ["x86_64"]},
             "VCpuInfo": {"DefaultVCpus": 8}, "MemoryInfo": {"SizeInMiB": 16384}},
        ]}

    def describe_instance_type_offerings(self, **kwargs):
        values = kwargs["Filters"][0]["Values"]
        return {"InstanceTypeOfferings": [{"InstanceType": item, "Location": "us-east-1"} for item in values]}


class FakeLogs:
    def __init__(self):
        self.requests = []
        self.published = []

    def put_log_events(self, **kwargs):
        self.published.append(kwargs)
        return {}

    def get_log_events(self, **kwargs):
        self.requests.append(kwargs)
        if len(self.requests) == 1:
            return {
                "events": [{"timestamp": 1, "message": "bootstrap ready"}],
                "nextForwardToken": "cursor-1",
            }
        return {
            "events": [{"timestamp": 2, "message": "building base"}],
            "nextForwardToken": "cursor-2",
        }


class FakeS3Paginator:
    def __init__(self, pages):
        self.pages = pages
        self.requests = []

    def paginate(self, **kwargs):
        self.requests.append(kwargs)
        return self.pages


class FakeS3:
    def __init__(self, pages):
        self.paginator = FakeS3Paginator(pages)
        self.deletions = []

    def get_paginator(self, name):
        if name != "list_object_versions":
            raise AssertionError(name)
        return self.paginator

    def delete_objects(self, **kwargs):
        self.deletions.append(kwargs)
        return {}


class FakeConsoleEc2:
    def __init__(self):
        self.output = "cloud-init starting\n"

    def get_console_output(self, **_kwargs):
        return {"Output": self.output}


class FakeHealthEc2:
    def describe_instance_status(self, **kwargs):
        self.request = kwargs
        return {"InstanceStatuses": [{
            "InstanceStatus": {"Status": "ok"},
            "SystemStatus": {"Status": "initializing"},
        }]}


class EventuallyConsistentEc2:
    def __init__(self):
        self.calls = 0

    def describe_instances(self, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            error = RuntimeError("not propagated")
            error.response = {"Error": {"Code": "InvalidInstanceID.NotFound"}}
            raise error
        return {"Reservations": [{"Instances": [{"InstanceId": "i-eventual", "State": {"Name": "pending"}}]}]}


class ReleaseValidationTest(unittest.TestCase):
    def shard(self):
        return {
            "id": "linux-test", "instanceType": "c7i.24xlarge",
            "amiQuery": {
                "owners": ["099720109477"], "ownerIds": ["099720109477"],
                "name": "ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*",
                "architecture": "x86_64",
            },
        }

    def test_resolve_ami_verifies_owner_architecture_and_storage(self):
        self.assertEqual("ami-verified", release.resolve_ami(FakeEc2(), FakeSsm(), self.shard()))

    def test_resolve_ami_rejects_wrong_owner(self):
        image = dict(FakeEc2().image, OwnerId="111111111111")
        with self.assertRaisesRegex(RuntimeError, "owned by"):
            release.resolve_ami(FakeEc2(image=image), FakeSsm(), self.shard())

    def test_matrix_rejects_instance_architecture_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, "does not support"):
            release.validate_launch_matrix(FakeEc2(architectures=["arm64"]), FakeSsm(), [self.shard()], "us-east-1")

    def test_matrix_rejects_unavailable_instance_type_in_zone(self):
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            release.validate_launch_matrix(FakeEc2(offered=False), FakeSsm(), [self.shard()], "us-east-1", "us-east-1a")

    def test_just_launched_instance_is_retried_during_ec2_eventual_consistency(self):
        ec2 = EventuallyConsistentEc2()
        with patch.object(release.time, "sleep") as sleep:
            instance = release.describe_instance_eventually(ec2, "i-eventual")
        self.assertEqual("i-eventual", instance["InstanceId"])
        self.assertEqual(2, ec2.calls)
        sleep.assert_called_once_with(2)

    def test_checked_in_plan_has_verification_for_every_shard(self):
        plan = release.load_plan(Path(__file__).with_name("release-plan.json"))
        self.assertTrue(plan["shards"])
        for shard in plan["shards"]:
            self.assertIn("amiQuery", shard)
            self.assertNotIn("amiSsmParameter", shard)

    def test_windows_shell_commands_use_explicit_git_bash(self):
        env = {"DL4J_BASH_EXE": r"C:\\Program Files\\Git\\bin\\bash.exe"}
        with patch.object(build_platform.platform, "system", return_value="Windows"):
            command = build_platform.bash_command(
                ["./update-versions.sh", "snapshot", "release"], env
            )
        self.assertEqual(env["DL4J_BASH_EXE"], command[0])
        self.assertEqual(
            ["./update-versions.sh", "snapshot", "release"], command[1:]
        )

    def test_managed_llvm_environment_carries_host_generators_and_existing_args(self):
        env = {"DL4J_CMAKE_ARGS": "-DEXISTING=ON"}
        build_platform._activate_managed_llvm_environment(
            env, "/cache/target-llvm", "/cache/host-tools/bin"
        )
        self.assertEqual("/cache/target-llvm", env["SD_TRITON_MANAGED_LLVM_ROOT"])
        self.assertEqual(
            "/cache/host-tools/bin", env["SD_TRITON_MANAGED_LLVM_HOST_TOOLS"]
        )
        self.assertIn("-DEXISTING=ON", env["DL4J_CMAKE_ARGS"])
        self.assertIn(
            "-DSD_TRITON_MANAGED_LLVM_ROOT=/cache/target-llvm",
            env["DL4J_CMAKE_ARGS"],
        )
        self.assertIn(
            "-DSD_TRITON_MANAGED_LLVM_HOST_TOOLS=/cache/host-tools/bin",
            env["DL4J_CMAKE_ARGS"],
        )

    def test_cross_triton_has_managed_host_tools_and_cold_build_fallback(self):
        dependencies = (
            Path(__file__).parents[2] / "libnd4j/cmake/Dependencies.cmake"
        ).read_text(encoding="utf-8")
        self.assertIn("SD_TRITON_MANAGED_LLVM_HOST_TOOLS", dependencies)
        self.assertIn("_TRITON_MANAGED_HOST_TOOLS_READY", dependencies)
        self.assertIn("NOT _TRITON_LLVM_INSTALL_COMPLETE OR", dependencies)
        self.assertIn("NOT _TRITON_COMPILER_INSTALL_COMPLETE", dependencies)
        self.assertIn('set(_TRITON_LLVM_RECIPE_REVISION "managed-llvm-patches-v12")', dependencies)
        self.assertIn('set(_TRITON_COMPILER_RECIPE_REVISION "managed-llvm-patches-v12")', dependencies)
        self.assertIn("managed-llvm-host-tools-v1", dependencies)
        self.assertIn("managed-sleef-host-tools-v1", dependencies)
        self.assertIn('"triton_llvm_host_tools"', dependencies)
        self.assertIn('"triton_sleef_host_tools"', dependencies)
        self.assertIn("sd_dep_cache_host_key(", dependencies)
        self.assertIn("dep_cache_store_${dep_name}_${_store_script_identity}.cmake", dependencies)

    def test_sccache_release_assets_are_pinned_to_quoted_response_file_support(self):
        self.assertEqual("0.17.0", build_platform.SCCACHE_VERSION)
        self.assertEqual(
            {
                ("linux", "x86_64"): (
                    "x86_64-unknown-linux-musl",
                    "67c4a96dd237c1f518f6b36083f270f9976d516f1e57fce891755ea782e50006",
                ),
                ("linux", "arm64"): (
                    "aarch64-unknown-linux-musl",
                    "821a86343191aa1cbab74bd42f9e93c9a63bf85e4742945f40d3ae84193c1c77",
                ),
                ("macos", "x86_64"): (
                    "x86_64-apple-darwin",
                    "c2144cafbfe3d22e34ae637f9974ce53613543ac19477fdb287df22ea3668261",
                ),
                ("macos", "arm64"): (
                    "aarch64-apple-darwin",
                    "0c560bfba31aef5bdfb4fb3d2677f6e61d71c5c00952f2a83344f47aa31f00f1",
                ),
                ("windows", "x86_64"): (
                    "x86_64-pc-windows-msvc",
                    "caf1932d76a909c909b7a2e41443cdfe3c79a49a380da1a22fa422e1d00d3ca7",
                ),
            },
            build_platform.SCCACHE_ASSETS,
        )
        root = Path(__file__).parents[2]
        pinned_files = [
            root / "release/aws/worker.ps1",
            root / "release/azure/worker.ps1",
            root / "release/gcp/worker.ps1",
            root / ".github/actions/setup-sccache-linux/action.yml",
            root / ".github/actions/setup-sccache-macos/action.yml",
            root / ".github/actions/setup-sccache-windows/action.yml",
        ]
        for pinned_file in pinned_files:
            source = pinned_file.read_text(encoding="utf-8")
            self.assertIn("v0.17.0", source, pinned_file)
            self.assertNotIn("v0.15.0", source, pinned_file)

    def test_cloud_plans_do_not_model_onednn_as_an_isa_extension(self):
        root = Path(__file__).parents[2]
        for provider in ("aws", "azure", "gcp"):
            plan = json.loads(
                (root / f"release/{provider}/release-plan.json").read_text(encoding="utf-8")
            )
            shard = next(item for item in plan["shards"] if item["id"] == "windows-x86_64-cpu")
            variant = next(item for item in shard["build"]["variants"] if item["name"] == "onednn")
            self.assertEqual("onednn", variant["helper"], provider)
            self.assertEqual("-onednn", variant["suffix"], provider)
            self.assertNotIn("extension", variant, provider)

    def test_desktop_vulkan_release_contract_is_portable_and_complete(self):
        root = Path(__file__).parents[2]
        for provider in ("aws", "azure", "gcp"):
            plan = json.loads(
                (root / f"release/{provider}/release-plan.json").read_text(
                    encoding="utf-8"
                )
            )
            for shard_id, platform in (
                ("linux-x86_64-vulkan", "linux-x86_64"),
                ("windows-x86_64-vulkan", "windows-x86_64"),
            ):
                shard = next(
                    item for item in plan["shards"] if item["id"] == shard_id
                )
                self.assertEqual("vulkan", shard["build"]["backend"], provider)
                self.assertEqual(platform, shard["build"]["javacppPlatform"], provider)
                self.assertEqual(["base"],
                                 [variant["name"] for variant in shard["build"]["variants"]],
                                 provider)
                if shard["os"] == "linux":
                    self.assertTrue(shard["build"]["variants"][0]["mlir"], provider)
                    self.assertTrue(shard["build"]["variants"][0]["triton"], provider)
                self.assertEqual(
                    {
                        "nd4j-vulkan",
                        "nd4j-vulkan-preset",
                        "nd4j-vulkan-platform",
                    },
                    set(shard["artifactRules"]["artifactIds"]),
                    provider,
                )
                self.assertEqual(
                    {
                        "nd4j-vulkan",
                        "nd4j-vulkan-preset",
                        "nd4j-vulkan-platform",
                    },
                    set(shard["artifactRules"]["unclassifiedArtifactIds"]),
                    provider,
                )
                self.assertIn(
                    f"-Dplatform.classifier={platform}",
                    shard["build"]["mavenArgs"],
                    provider,
                )
                if shard["os"] == "windows":
                    self.assertFalse(
                        any(
                            variant.get("mlir") or variant.get("triton")
                            for variant in shard["build"]["variants"]
                        ),
                        provider,
                    )

        for worker in (
            root / "release/aws/worker.ps1",
            root / "release/azure/worker.ps1",
            root / "release/gcp/worker.ps1",
        ):
            source = worker.read_text(encoding="utf-8")
            self.assertIn("mingw-w64-x86_64-vulkan-headers", source, worker)
            self.assertIn("mingw-w64-x86_64-vulkan-loader", source, worker)

        action_source = (root / ".github/actions/run-release-worker/action.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("mingw-w64-x86_64-vulkan-headers", action_source)
        self.assertIn("mingw-w64-x86_64-vulkan-loader", action_source)

    def test_sdx_classifier_tokens_include_backend_specific_runtime_jars(self):
        build = {
            "backend": "vulkan",
            "profiles": ["sdx"],
            "javacppPlatform": "linux-x86_64",
            "variants": [
                {"name": "compile", "platformExtension": "-compile"}
            ],
        }
        rules = {
            "mode": "classifier",
            "artifactIds": ["nd4j-vulkan"],
            "classifierTokens": ["linux-x86_64-compile"],
        }

        build_platform.enable_sdx_release_component(build, rules)

        self.assertIn("nd4j-sdx", rules["artifactIds"])
        self.assertIn("linux-x86_64-vulkan-compile", rules["classifierTokens"])

    def test_smoke_overrides_instance_and_build_threads(self):
        shard = self.shard()
        shard["build"] = {"buildThreads": 48}
        release.apply_execution_overrides([shard], "c7i.xlarge", 4)
        self.assertEqual("c7i.xlarge", shard["instanceType"])
        self.assertEqual(4, shard["build"]["buildThreads"])

    def test_smoke_override_rejects_accelerator_instances(self):
        with self.assertRaisesRegex(SystemExit, "invalid CPU compile"):
            release.apply_execution_overrides([self.shard()], "g6.xlarge", 4)

    def test_matrix_uses_serial_reusable_lanes(self):
        plan = release.load_plan(Path(__file__).with_name("release-plan.json"))
        lanes = release.execution_shards(plan)
        self.assertEqual(len(plan["shards"]), len(lanes))
        linux = next(item for item in lanes if item["id"] == "linux-x86_64-cpu")
        self.assertEqual(9, len(linux["build"]["variants"]))
        self.assertNotIn("--base", linux["id"])

    def test_windows_shards_omit_unsupported_managed_llvm_variants(self):
        plan = release.load_plan(Path(__file__).with_name("release-plan.json"))
        windows = [item for item in plan["shards"] if item["os"] == "windows"]
        self.assertTrue(windows)
        for shard in windows:
            variants = shard["build"]["variants"]
            compile_variants = [item for item in variants if item["name"] == "compile"]
            if shard["id"] == "windows-x86_64-cpu" or (
                shard["build"]["backend"] == "cuda" and not shard["build"].get("zludaVersion")
            ):
                self.assertEqual(1, len(compile_variants), shard["id"])
                self.assertTrue(compile_variants[0].get("windowsNativeCompile"))
                self.assertEqual("compile", compile_variants[0].get("extension"))
                self.assertEqual("-compile", compile_variants[0].get("suffix"))
                if shard["build"]["backend"] == "cuda":
                    self.assertEqual(
                        f"-cuda-{shard['build']['cudaVersion']}-compile",
                        compile_variants[0].get("classifierSuffix"),
                    )
                    self.assertEqual("-compile", compile_variants[0].get("platformExtension"))
            else:
                self.assertFalse(compile_variants, shard["id"])
            self.assertFalse(
                any(item.get("mlir") or item.get("triton") for item in variants),
                shard["id"],
            )
        linux = next(item for item in plan["shards"] if item["id"] == "linux-x86_64-cpu")
        self.assertIn("compile", [item["name"] for item in linux["build"]["variants"]])

    def test_plan_rejects_windows_managed_llvm_before_provisioning(self):
        for variant in (
            {"name": "compile", "mlir": True},
            {"name": "cuda-compile", "triton": True},
        ):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as temp:
                plan = json.loads(
                    Path(__file__).with_name("release-plan.json").read_text(encoding="utf-8")
                )
                windows = next(item for item in plan["shards"] if item["os"] == "windows")
                windows["build"]["variants"].append(variant)
                path = Path(temp) / "plan.json"
                path.write_text(json.dumps(plan), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "unsupported by MSVC"):
                    release.load_plan(path)

    def test_s3_compiler_cache_uses_a_stable_cross_run_namespace(self):
        plan = {"artifactPrefix": "deeplearning4j/releases"}
        cache = release.compiler_cache_config(plan, "release-bucket", "us-east-2")
        self.assertEqual({
            "backend": "s3",
            "bucket": "release-bucket",
            "region": "us-east-2",
            "keyPrefix": "deeplearning4j/releases/compiler-cache/v1",
        }, cache)
        self.assertNotIn("runId", cache["keyPrefix"])

    def test_instance_role_can_read_and_publish_sccache_objects(self):
        class ClientError(Exception):
            pass

        class Iam:
            def __init__(self):
                self.policy = None

            def get_role(self, **_kwargs):
                return {"Role": {}}

            def put_role_policy(self, **kwargs):
                self.policy = json.loads(kwargs["PolicyDocument"])

            def get_instance_profile(self, **_kwargs):
                return {"InstanceProfile": {"Roles": [{"RoleName": "DL4JReleaseBuilderRole"}]}}

        iam = Iam()
        with patch.object(release, "_boto3", return_value=(object(), ClientError)):
            release.ensure_instance_profile(iam, "release-bucket", "kill", "/logs")
        s3_actions = set(iam.policy["Statement"][0]["Action"])
        self.assertTrue({"s3:GetObject", "s3:PutObject", "s3:ListBucket"}.issubset(s3_actions))

    def test_shared_driver_configures_two_level_remote_sccache_backends(self):
        settings = {
            "s3": {
                "backend": "s3", "bucket": "s3-bucket", "region": "us-east-2",
                "keyPrefix": "cache/v1",
            },
            "gcs": {
                "backend": "gcs", "bucket": "gcs-bucket", "keyPrefix": "cache/v1",
            },
            "azure": {
                "backend": "azure", "container": "releases", "keyPrefix": "cache/v1",
                "connectionString": "BlobEndpoint=https://account/;SharedAccessSignature=token",
            },
        }
        for backend, remote in settings.items():
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as temp:
                source = Path(temp) / "source"
                source.mkdir()
                environment = {"PATH": "/existing/bin"}
                with patch.object(
                    build_platform, "ensure_sccache", return_value="/tools/sccache"
                ), patch.object(build_platform, "run") as execute:
                    executable, started = build_platform.configure_compiler_cache(
                        {"compilerCache": remote}, source, environment
                    )
                self.assertEqual("/tools/sccache", executable)
                self.assertTrue(started)
                self.assertEqual(f"disk,{backend}", environment["SCCACHE_MULTILEVEL_CHAIN"])
                self.assertEqual("all", environment["SCCACHE_MULTILEVEL_WRITE_ERROR_POLICY"])
                self.assertEqual(str(source.resolve()), environment["SCCACHE_BASEDIRS"])
                self.assertEqual("1", environment["SD_USE_SCCACHE"])
                self.assertEqual("ON", environment["SD_REQUIRE_COMPILER_CACHE"])
                self.assertEqual("/tools/sccache", environment["DL4J_COMPILER_CACHE"])
                self.assertEqual(
                    ["/tools", "/existing/bin"], environment["PATH"].split(os.pathsep)
                )
                for launcher in (
                    "CMAKE_C_COMPILER_LAUNCHER",
                    "CMAKE_CXX_COMPILER_LAUNCHER",
                    "CMAKE_CUDA_COMPILER_LAUNCHER",
                ):
                    self.assertEqual("/tools/sccache", environment[launcher])
                execute.assert_called_once_with(
                    ["/tools/sccache", "--start-server"], source, environment
                )
        self.assertEqual("s3-bucket", settings["s3"]["bucket"])

    def test_shared_driver_reads_azure_sccache_secret_from_environment(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            source.mkdir()
            environment = {
                "PATH": "/existing/bin",
                "GITHUB_AZURE_CACHE_SECRET": "BlobEndpoint=https://account/;token",
            }
            remote = {
                "backend": "azure",
                "container": "releases",
                "keyPrefix": "cache/v1",
                "connectionStringEnv": "GITHUB_AZURE_CACHE_SECRET",
            }
            with patch.object(
                build_platform, "ensure_cached_sccache", return_value="/tools/sccache"
            ), patch.object(build_platform, "run"):
                build_platform.configure_compiler_cache(
                    {"compilerCache": remote}, source, environment
                )
            self.assertEqual(
                "BlobEndpoint=https://account;token",
                environment["SCCACHE_AZURE_CONNECTION_STRING"],
            )
            self.assertNotIn("connectionString", remote)

    def test_shared_driver_restores_prebuilt_libnd4j_archive(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            archive = root / "libnd4j.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("payload/libnd4j/blasbuild/cpu/libnd4jcpu.so", b"native")
                bundle.writestr("payload/libnd4j/blasbuild/cpu/blas/libopenblas.so", b"blas")
            openblas = root / "openblas"
            environment = {"OPENBLAS_PATH": str(openblas)}

            def copy_archive(_url, destination, _description):
                destination.write_bytes(archive.read_bytes())

            with patch.object(build_platform, "download_with_retry", side_effect=copy_archive):
                url = build_platform.prepare_prebuilt_libnd4j(
                    source,
                    {"backend": "cpu", "libnd4jUrl": "https://example.invalid/libnd4j.zip"},
                    environment,
                )

            self.assertEqual("https://example.invalid/libnd4j.zip", url)
            self.assertEqual(
                b"native",
                (source / "libnd4j/blasbuild/cpu/libnd4jcpu.so").read_bytes(),
            )
            self.assertIn(
                "DEFAULT_ENGINE samediff::ENGINE_CPU",
                (source / "libnd4j/include/config.h").read_text(),
            )
            self.assertTrue((source / "libnd4j/include/generated/include_ops.h").is_file())
            self.assertEqual(b"blas", (openblas / "lib/libopenblas.so").read_bytes())

    def test_shared_driver_prefetches_and_publishes_one_l0_working_set(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            source.mkdir()
            cache_dir = Path(temp) / "mounted-sccache" / "cache"
            environment = {
                "PATH": "/existing/bin",
                "DL4J_CLOUD_IO": "/cloud-io.py",
                "DL4J_DEPENDENCY_CACHE_HELPER": "/dependency-cache.py",
                "DL4J_SCCACHE_DIR": str(cache_dir),
            }
            config = {
                "commit": "commit-a",
                "managedIdentityClientId": "identity",
                "shard": {"id": "linux-zluda", "contractDigest": "contract-a"},
                "compilerCache": {
                    "backend": "azure",
                    "account": "account",
                    "container": "releases",
                    "keyPrefix": "compiler/v1",
                    "connectionString": "BlobEndpoint=https://account/;SharedAccessSignature=sas",
                    "toolchainCache": {"schemaVersion": 1, "keyPrefix": "tools/v1"},
                    "localSnapshot": {"schemaVersion": 1, "name": "sccache-l0"},
                },
            }
            transport = (Path("/dependency-cache.py"), "account/releases", "tools/v1", "identity")
            with patch.object(
                build_platform, "toolchain_cache_transport", return_value=transport
            ), patch.object(
                build_platform, "restore_toolchain_dependency", return_value=False
            ) as restore, patch.object(
                build_platform, "ensure_cached_sccache", return_value="/tools/sccache"
            ), patch.object(build_platform, "run"):
                executable, started = build_platform.configure_compiler_cache(
                    config, source, environment
                )
            self.assertEqual("/tools/sccache", executable)
            self.assertTrue(started)
            identity = environment["DL4J_SCCACHE_SNAPSHOT_IDENTITY"]
            restore.assert_called_once_with(
                config,
                environment,
                name="sccache-l0",
                identity=identity,
                destination=cache_dir,
            )
            self.assertEqual("false", environment["DL4J_SCCACHE_SNAPSHOT_RESTORED"])

            cache_file = cache_dir / "object"
            cache_file.write_bytes(b"cached-object")
            with patch.object(build_platform, "publish_toolchain_dependency") as publish:
                metrics = build_platform.publish_compiler_cache_snapshot(
                    config, environment
                )
            publish.assert_called_once_with(
                config,
                environment,
                name="sccache-l0",
                identity=identity,
                source=cache_dir,
            )
            self.assertEqual("published", metrics["publishStatus"])
            self.assertGreaterEqual(metrics["expandedBytes"], len(b"cached-object"))

            same_contract = json.loads(json.dumps(config))
            same_contract["commit"] = "commit-b"
            self.assertEqual(
                identity,
                build_platform.compiler_cache_snapshot_identity(same_contract),
            )
            changed_contract = json.loads(json.dumps(config))
            changed_contract["shard"]["contractDigest"] = "contract-b"
            self.assertNotEqual(
                identity,
                build_platform.compiler_cache_snapshot_identity(changed_contract),
            )

    def test_restored_l0_snapshot_is_not_republished(self):
        environment = {
            "DL4J_SCCACHE_SNAPSHOT_IDENTITY": "a" * 64,
            "DL4J_SCCACHE_SNAPSHOT_DIR": "/cache",
            "DL4J_SCCACHE_SNAPSHOT_RESTORED": "true",
            "DL4J_SCCACHE_SNAPSHOT_RESTORE_SECONDS": "1.25",
        }
        with patch.object(build_platform, "publish_toolchain_dependency") as publish:
            metrics = build_platform.publish_compiler_cache_snapshot({}, environment)
        publish.assert_not_called()
        self.assertEqual("hit", metrics["restoreStatus"])
        self.assertEqual("not-required", metrics["publishStatus"])

    def test_shared_driver_activates_local_sccache_but_not_ccache(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            source.mkdir()

            sccache_env = {"PATH": "/existing/bin"}
            with patch.object(
                build_platform.shutil,
                "which",
                side_effect=lambda name: "/tools/sccache" if name == "sccache" else None,
            ), patch.object(build_platform, "run") as execute:
                executable, started = build_platform.configure_compiler_cache(
                    {}, source, sccache_env
                )
            self.assertEqual("/tools/sccache", executable)
            self.assertTrue(started)
            self.assertEqual("1", sccache_env["SD_USE_SCCACHE"])
            self.assertEqual("ON", sccache_env["SD_REQUIRE_COMPILER_CACHE"])
            self.assertEqual("/tools/sccache", sccache_env["DL4J_COMPILER_CACHE"])
            self.assertEqual(
                ["/tools", "/existing/bin"], sccache_env["PATH"].split(os.pathsep)
            )
            execute.assert_called_once_with(
                ["/tools/sccache", "--start-server"], source, sccache_env
            )

            ccache_env = {"PATH": "/existing/bin"}
            with patch.object(
                build_platform.shutil,
                "which",
                side_effect=lambda name: "/tools/ccache" if name == "ccache" else None,
            ), patch.object(build_platform, "run") as execute:
                executable, started = build_platform.configure_compiler_cache(
                    {}, source, ccache_env
                )
            self.assertEqual("/tools/ccache", executable)
            self.assertFalse(started)
            self.assertNotIn("SD_USE_SCCACHE", ccache_env)
            self.assertEqual("ON", ccache_env["SD_REQUIRE_COMPILER_CACHE"])
            self.assertEqual("/tools/ccache", ccache_env["DL4J_COMPILER_CACHE"])
            self.assertEqual("/existing/bin", ccache_env["PATH"])
            execute.assert_called_once_with(
                ["/tools/ccache", "--zero-stats"], source, ccache_env
            )

    def test_shared_driver_rejects_incomplete_remote_cache_configuration(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            source.mkdir()
            with self.assertRaisesRegex(ValueError, "compilerCache.region"):
                with patch.object(build_platform, "ensure_sccache", return_value="sccache"):
                    build_platform.configure_compiler_cache(
                        {"compilerCache": {
                            "backend": "s3", "bucket": "bucket", "keyPrefix": "cache/v1",
                        }},
                        source,
                        {},
                    )

    def test_expanded_smoke_selector_keeps_single_variant(self):
        plan = release.load_plan(Path(__file__).with_name("release-plan.json"))
        selected = release.selected_executions(plan, ["linux-x86_64-cpu--base"])
        self.assertEqual(1, len(selected))
        self.assertEqual("linux-x86_64-cpu--base", selected[0]["id"])
        self.assertEqual(["base"], [item["name"] for item in selected[0]["build"]["variants"]])

    def test_core_constraint_greedily_selects_largest_feasible_size(self):
        lane = self.shard()
        lane["build"] = {"buildThreads": 48, "mavenHeapGiB": 32}
        schedule = release.apply_core_constraint(SizingEc2(), [lane], 5)
        self.assertEqual("c7i.xlarge", lane["instanceType"])
        self.assertEqual(4, lane["build"]["buildThreads"])
        self.assertEqual(4, lane["build"]["mavenHeapGiB"])
        self.assertEqual(4, schedule[0]["selectedVcpus"])

    def test_core_constraint_fails_before_launch_when_no_size_fits(self):
        lane = self.shard()
        lane["build"] = {"buildThreads": 48, "mavenHeapGiB": 32}
        with self.assertRaisesRegex(SystemExit, "Core constraint is infeasible"):
            release.apply_core_constraint(SizingEc2(), [lane], 1)

    def test_s3_log_purge_deletes_only_selected_build_log_versions(self):
        s3 = FakeS3([{
            "Versions": [
                {"Key": "deeplearning4j/releases/run-a/linux-cpu--base/build.log", "VersionId": "v1"},
                {"Key": "deeplearning4j/releases/run-a/linux-cpu/status.json", "VersionId": "v2"},
                {"Key": "deeplearning4j/releases/run-a/windows-cpu/build.log", "VersionId": "v3"},
            ],
            "DeleteMarkers": [
                {"Key": "deeplearning4j/releases/run-a/linux-cpu--base/build.log", "VersionId": "d1"},
            ],
        }])
        deleted = release.delete_s3_log_objects(
            s3, "release-bucket", "deeplearning4j/releases/run-a/", {"linux-cpu"},
        )
        self.assertEqual(["v1", "d1"], [item["VersionId"] for item in deleted])
        self.assertEqual("deeplearning4j/releases/run-a/", s3.paginator.requests[0]["Prefix"])
        request_objects = s3.deletions[0]["Delete"]["Objects"]
        self.assertEqual(deleted, request_objects)
        self.assertFalse(any(item["Key"].endswith("status.json") for item in request_objects))
        self.assertTrue(release.shard_selected("linux-cpu--avx2", {"linux-cpu"}))
        self.assertFalse(release.shard_selected("windows-cpu", {"linux-cpu"}))

    def test_s3_log_purge_batches_at_aws_limit(self):
        versions = [
            {"Key": f"deeplearning4j/releases/run-{index}/linux-cpu/build.log", "VersionId": str(index)}
            for index in range(1001)
        ]
        s3 = FakeS3([{"Versions": versions}])
        deleted = release.delete_s3_log_objects(s3, "release-bucket", "deeplearning4j/releases/")
        self.assertEqual(1001, len(deleted))
        self.assertEqual([1000, 1], [len(call["Delete"]["Objects"]) for call in s3.deletions])

    def test_controller_event_is_published_without_worker_bootstrap(self):
        logs = FakeLogs()
        self.assertTrue(release.emit_cloudwatch_event(logs, "/releases", "run/lane", "phase=launched"))
        self.assertEqual("/releases", logs.published[0]["logGroupName"])
        self.assertEqual("run/lane", logs.published[0]["logStreamName"])
        self.assertIn("[dl4j-controller] phase=launched", logs.published[0]["logEvents"][0]["message"])

    def test_stream_lane_logs_prints_events_and_reuses_cursor(self):
        logs = FakeLogs()
        output = StringIO()
        with redirect_stdout(output):
            token, count = release.stream_lane_logs(logs, "/releases", "run/lane")
            token, second_count = release.stream_lane_logs(logs, "/releases", "run/lane", token)
        self.assertEqual("cursor-2", token)
        self.assertEqual((1, 1), (count, second_count))
        self.assertEqual("cursor-1", logs.requests[1]["nextToken"])
        self.assertIn("[run/lane] bootstrap ready", output.getvalue())
        self.assertIn("[run/lane] building base", output.getvalue())

    def test_stream_console_output_only_prints_new_bytes(self):
        ec2 = FakeConsoleEc2()
        output = StringIO()
        with redirect_stdout(output):
            offset = release.stream_console_output(ec2, "i-test", 0)
            ec2.output += "worker downloaded\n"
            offset = release.stream_console_output(ec2, "i-test", offset)
        self.assertEqual(len(ec2.output), offset)
        self.assertEqual(1, output.getvalue().count("cloud-init starting"))
        self.assertEqual(1, output.getvalue().count("worker downloaded"))

    def test_instance_health_includes_aws_system_and_instance_checks(self):
        ec2 = FakeHealthEc2()
        self.assertEqual(("ok", "initializing"), release.instance_health(ec2, "i-test"))
        self.assertTrue(ec2.request["IncludeAllInstances"])

    def test_log_forwarder_supports_ubuntu_aws_cli_v1(self):
        with patch.object(log_forwarder.subprocess, "run") as run:
            log_forwarder.aws("us-east-1", "logs", "create-log-stream")
        command = run.call_args.args[0]
        self.assertNotIn("--no-cli-pager", command)
        self.assertEqual("", run.call_args.kwargs["env"]["AWS_PAGER"])

    def test_failed_lane_log_is_retrieved_from_s3(self):
        class LogS3:
            request = None

            def get_object(self, **kwargs):
                self.request = kwargs
                return {"Body": BytesIO(b"phase=rust-toolchain status=failed\nreal bootstrap error\n")}

        s3 = LogS3()
        output = StringIO()
        with redirect_stdout(output):
            retrieved = release.print_s3_build_log(
                s3, "release-bucket", "runs/run-1/lane/status.json", "lane"
            )
        self.assertTrue(retrieved)
        self.assertEqual("runs/run-1/lane/build.log", s3.request["Key"])
        self.assertEqual("bytes=-262144", s3.request["Range"])
        self.assertIn("real bootstrap error", output.getvalue())

    def test_native_lane_invokes_shared_github_script_before_cross_platform_script(self):
        build = {
            "javacppPlatform": "linux-x86_64", "backend": "cpu", "profiles": ["cpu", "sdx"],
            "modules": [":libnd4j"], "variants": [{"name": "base"}], "buildCrossPlatform": True,
        }
        shard = {"id": "linux-x86_64-cpu", "os": "linux", "build": build}
        events = []
        with patch.object(build_platform, "prepare_openblas"), \
                patch.object(build_platform, "run", side_effect=lambda command, *_args: events.append(Path(command[1]).name)), \
                patch.object(build_platform, "build_cross_platform", side_effect=lambda *_args: events.append("cross-platform.sh")):
            build_platform.build_native_platform(Path("/source"), shard, Path("/m2"), {}, None)
        self.assertEqual(["linux-x86_64.sh", "cross-platform.sh"], events)

    def test_classifier_only_native_lane_skips_cross_platform_java_reactor(self):
        build = {
            "javacppPlatform": "linux-x86_64",
            "backend": "cpu",
            "profiles": ["cpu", "sdx"],
            "modules": [":libnd4j"],
            "variants": [{"name": "compile", "suffix": "-compile", "mlir": True}],
            "buildCrossPlatform": True,
        }
        shard = {"id": "linux-x86_64-cpu", "os": "linux", "build": build}
        events = []
        with patch.object(build_platform, "prepare_openblas"), \
                patch.object(build_platform, "run", side_effect=lambda command, *_args: events.append(Path(command[1]).name)), \
                patch.object(build_platform, "build_cross_platform", side_effect=lambda *_args: events.append("cross-platform.sh")):
            build_platform.build_native_platform(Path("/source"), shard, Path("/m2"), {}, None)
        self.assertEqual(["linux-x86_64.sh"], events)

    def test_classifier_only_native_lane_skips_base_aot_package(self):
        build = {
            "javacppPlatform": "linux-x86_64",
            "backend": "cpu",
            "variants": [{"name": "compile", "suffix": "-compile", "mlir": True}],
            "buildAot": True,
        }
        with patch.object(build_platform, "run") as run:
            produced = build_platform.build_aot(
                Path("/source"), Path("/output"), build, Path("/m2"), {}
            )
        self.assertEqual(0, produced)
        run.assert_not_called()

    def test_native_lane_checkpoints_each_variant_before_a_later_failure(self):
        build = {
            "javacppPlatform": "linux-x86_64",
            "backend": "cpu",
            "profiles": ["cpu"],
            "modules": [":libnd4j"],
            "variants": [{"name": "base"}, {"name": "compile"}],
        }
        shard = {
            "id": "linux-x86_64-cpu",
            "os": "linux",
            "artifactRules": {},
            "build": build,
        }
        calls = []

        def run_variant(*_args):
            calls.append("run")
            if len(calls) == 2:
                raise RuntimeError("compile failed")

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            progress = root / "build-result.json"
            with patch.object(build_platform, "prepare_openblas"), patch.object(
                build_platform, "run", side_effect=run_variant
            ), patch.object(
                build_platform, "attest_variant_classifier_artifacts"
            ), patch.object(
                build_platform, "attest_unclassified_artifacts"
            ), patch.object(
                build_platform, "stage_repository"
            ) as stage:
                with self.assertRaisesRegex(RuntimeError, "compile failed"):
                    build_platform.build_native_platform(
                        Path("/source"),
                        shard,
                        root / "m2",
                        {},
                        None,
                        maven_output=root / "staged",
                        progress_output=progress,
                    )

            self.assertEqual(1, stage.call_count)
            self.assertEqual(
                ["base"],
                json.loads(progress.read_text(encoding="utf-8"))["completedVariants"],
            )

    def test_native_lane_records_variant_build_duration_on_failure(self):
        shard = {
            "id": "linux-x86_64-cpu",
            "os": "linux",
            "artifactRules": {},
            "build": {
                "javacppPlatform": "linux-x86_64",
                "backend": "cpu",
                "profiles": ["cpu"],
                "modules": [":libnd4j"],
                "variants": [{"name": "base"}],
            },
        }
        with tempfile.TemporaryDirectory() as temp:
            benchmark_path = Path(temp) / "build-benchmark.json"
            benchmark = {"schemaVersion": 1, "variants": []}
            with patch.object(build_platform, "prepare_openblas"), patch.object(
                build_platform, "run", side_effect=RuntimeError("native failed")
            ):
                with self.assertRaisesRegex(RuntimeError, "native failed"):
                    build_platform.build_native_platform(
                        Path("/source"),
                        shard,
                        Path(temp) / "m2",
                        {},
                        None,
                        benchmark=benchmark,
                        benchmark_output=benchmark_path,
                    )
            recorded = json.loads(benchmark_path.read_text(encoding="utf-8"))
            self.assertEqual("base", recorded["variants"][0]["name"])
            self.assertEqual("failed", recorded["variants"][0]["status"])
            self.assertGreaterEqual(recorded["variants"][0]["durationSeconds"], 0)

    def test_android_release_disables_host_jvm_linking_and_selects_variant_api(self):
        source = Path("/source")
        build = {"javacppPlatform": "android-arm64"}
        environment = {
            "ANDROID_NDK": "/opt/android/android-ndk-r27d",
            "OPENBLAS_PATH": "/opt/openblas",
        }
        base = build_platform.android_cmake_args(
            source, build, {"name": "base"}, environment
        )
        nnapi = build_platform.android_cmake_args(
            source, build, {"name": "compile-nnapi"}, environment
        )
        self.assertIn("-DSD_BUILD_WITH_JAVA=OFF", base)
        self.assertIn("-DANDROID_PLATFORM=android-21", base)
        self.assertIn("-DSD_BUILD_WITH_JAVA=OFF", nnapi)
        self.assertIn("-DANDROID_PLATFORM=android-27", nnapi)

    def test_android_release_driver_propagates_each_variant_api(self):
        shard = {
            "id": "android-arm64",
            "os": "linux",
            "artifactRules": {},
            "build": {
                "backend": "cpu",
                "javacppPlatform": "android-arm64",
                "buildThreads": 4,
                "variants": [{"name": "base"}, {"name": "compile-nnapi"}],
            },
        }
        environment = {
            "ANDROID_NDK": "/opt/android/android-ndk-r27d",
            "OPENBLAS_PATH": "/opt/openblas",
        }
        invocations = []
        with patch.object(build_platform, "prepare_openblas"), \
                patch.object(build_platform, "reset_variant_classifier_artifacts"), \
                patch.object(build_platform, "attest_variant_classifier_artifacts"), \
                patch.object(
                    build_platform,
                    "run",
                    side_effect=lambda _command, _source, env: invocations.append(env.copy()),
                ):
            build_platform.build_native_platform(
                Path("/source"), shard, Path("/m2"), environment, None
            )
        self.assertEqual(["21", "27"], [
            invocation["DL4J_ANDROID_API"] for invocation in invocations
        ])
        self.assertTrue(all(
            "-DSD_BUILD_WITH_JAVA=OFF" in invocation["DL4J_CMAKE_ARGS"]
            for invocation in invocations
        ))

    def test_every_covered_workflow_uses_a_shared_release_executor(self):
        repository_root = Path(__file__).resolve().parents[2]
        plan = json.loads((Path(__file__).with_name("release-plan.json")).read_text(encoding="utf-8"))
        shared = ".github/workflows/_release-worker.yml"
        missing = []
        for relative_path in plan["coveredWorkflows"]:
            workflow_path = Path(relative_path)
            if workflow_path.parent == Path("."):
                workflow_path = Path(".github/workflows") / workflow_path
            workflow = (repository_root / workflow_path).read_text(encoding="utf-8")
            if shared not in workflow:
                missing.append(relative_path)
        self.assertEqual([], missing)

    def test_every_accelerator_backend_is_profile_gated_like_cuda(self):
        repository_root = Path(__file__).resolve().parents[2]
        pom = ET.parse(repository_root / "nd4j/nd4j-backends/nd4j-backend-impls/pom.xml").getroot()
        namespace = {"m": "http://maven.apache.org/POM/4.0.0"}
        direct_modules = {node.text for node in pom.findall("m:modules/m:module", namespace)}
        expected_profiles = {
            "cuda": {"nd4j-cuda-backend-common", "nd4j-cuda", "nd4j-cuda-preset", "nd4j-cuda-platform"},
            "metal": {"nd4j-metal", "nd4j-metal-preset"},
            "tpu": {"nd4j-tpu", "nd4j-tpu-preset"},
            "hexagon": {"nd4j-hexagon", "nd4j-hexagon-preset"},
            "vulkan": {"nd4j-vulkan", "nd4j-vulkan-preset", "nd4j-vulkan-platform"},
            "zluda": {"nd4j-cuda-backend-common", "nd4j-zluda"},
            "zluda-platform": {"nd4j-zluda-platform"},
            "zluda-amd": {"nd4j-cuda-backend-common", "nd4j-zluda"},
        }
        profiles = {}
        for profile in pom.findall("m:profiles/m:profile", namespace):
            profile_id = profile.findtext("m:id", namespaces=namespace)
            modules = {node.text for node in profile.findall("m:modules/m:module", namespace)}
            active_by_default = profile.findtext("m:activation/m:activeByDefault", namespaces=namespace)
            profiles[profile_id] = (modules, active_by_default)

        accelerator_modules = set().union(*expected_profiles.values())
        self.assertTrue(direct_modules.isdisjoint(accelerator_modules))
        for profile_id, expected_modules in expected_profiles.items():
            self.assertIn(profile_id, profiles)
            self.assertEqual(expected_modules, profiles[profile_id][0])
            self.assertEqual("false", profiles[profile_id][1])

    def test_zluda_platform_selects_exactly_one_os_native_classifier(self):
        repository_root = Path(__file__).resolve().parents[2]
        namespace = {"m": "http://maven.apache.org/POM/4.0.0"}
        pom = ET.parse(
            repository_root
            / "nd4j/nd4j-backends/nd4j-backend-impls/nd4j-zluda-platform/pom.xml"
        ).getroot()

        self.assertEqual(
            "nd4j-zluda-12.9-platform",
            pom.findtext("m:artifactId", namespaces=namespace),
        )
        self.assertEqual(
            "false",
            pom.findtext("m:properties/m:skipPublishing", namespaces=namespace),
        )
        self.assertEqual(
            "12.9",
            pom.findtext("m:properties/m:cuda.version", namespaces=namespace),
        )
        self.assertEqual(
            "7.2.4",
            pom.findtext("m:properties/m:rocm.version", namespaces=namespace),
        )
        self.assertEqual(
            "nd4j-zluda-${cuda.version}",
            pom.findtext("m:properties/m:nd4j.backend", namespaces=namespace),
        )
        direct_dependencies = pom.findall("m:dependencies/m:dependency", namespace)
        self.assertEqual(1, len(direct_dependencies))
        self.assertEqual(
            "nd4j-zluda-12.9",
            direct_dependencies[0].findtext("m:artifactId", namespaces=namespace),
        )
        self.assertIsNone(
            direct_dependencies[0].findtext("m:classifier", namespaces=namespace)
        )

        expected_profiles = {
            "zluda-linux-amd64": ("Linux", None, "amd64", "${zluda.linux.classifier}"),
            "zluda-linux-x86_64": ("Linux", None, "x86_64", "${zluda.linux.classifier}"),
            "zluda-windows-amd64": (None, "windows", "amd64", "${zluda.windows.classifier}"),
            "zluda-windows-x86_64": (None, "windows", "x86_64", "${zluda.windows.classifier}"),
        }
        profiles = {
            profile.findtext("m:id", namespaces=namespace): profile
            for profile in pom.findall("m:profiles/m:profile", namespace)
        }
        self.assertEqual(set(expected_profiles), set(profiles))
        for profile_id, (name, family, arch, classifier) in expected_profiles.items():
            profile = profiles[profile_id]
            os_activation = profile.find("m:activation/m:os", namespace)
            self.assertIsNotNone(os_activation)
            self.assertEqual(name, os_activation.findtext("m:name", namespaces=namespace))
            self.assertEqual(family, os_activation.findtext("m:family", namespaces=namespace))
            self.assertEqual(arch, os_activation.findtext("m:arch", namespaces=namespace))
            dependencies = profile.findall("m:dependencies/m:dependency", namespace)
            self.assertEqual(1, len(dependencies))
            self.assertEqual(
                "${nd4j.backend}",
                dependencies[0].findtext("m:artifactId", namespaces=namespace),
            )
            self.assertEqual(
                classifier,
                dependencies[0].findtext("m:classifier", namespaces=namespace),
            )

        all_artifact_ids = {
            dependency.findtext("m:artifactId", namespaces=namespace)
            for dependency in pom.findall(".//m:dependency", namespace)
        }
        self.assertNotIn("nd4j-cuda-12.9-platform", all_artifact_ids)
        self.assertNotIn("cuda-platform", all_artifact_ids)

    def test_every_zluda_shard_publishes_cuda_versioned_maven_coordinates(self):
        repository_root = Path(__file__).resolve().parents[2]
        for provider in ("aws", "azure", "gcp"):
            plan = json.loads(
                (repository_root / f"release/{provider}/release-plan.json").read_text(encoding="utf-8")
            )
            zluda_shards = [
                shard for shard in plan["shards"] if shard["build"].get("zludaVersion")
            ]
            self.assertTrue(zluda_shards, provider)
            for shard in zluda_shards:
                cuda_version = shard["build"]["cudaVersion"]
                backend = f"nd4j-zluda-{cuda_version}"
                platform = f"{backend}-platform"
                rules = shard["artifactRules"]
                self.assertIn(backend, rules["artifactIds"], shard["id"])
                self.assertIn(platform, rules["artifactIds"], shard["id"])
                self.assertIn(backend, rules["unclassifiedArtifactIds"], shard["id"])
                self.assertIn(platform, rules["unclassifiedArtifactIds"], shard["id"])
                self.assertIn(f":{backend}", shard["build"]["modules"], shard["id"])
                self.assertIn(f":{platform}", shard["build"]["modules"], shard["id"])
                self.assertNotIn(
                    f":nd4j-cuda-{cuda_version}", shard["build"]["modules"], shard["id"]
                )
                self.assertNotIn(
                    f"nd4j-cuda-{cuda_version}", rules["artifactIds"], shard["id"]
                )

    def test_no_pom_has_duplicate_top_level_profiles_sections(self):
        repository_root = Path(__file__).resolve().parents[2]
        namespace = {"m": "http://maven.apache.org/POM/4.0.0"}
        tracked_files = subprocess.run(
            ["git", "ls-files"], cwd=repository_root, check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        duplicates = []
        for relative_path in tracked_files:
            if Path(relative_path).name != "pom.xml":
                continue
            pom = ET.parse(repository_root / relative_path).getroot()
            count = len(pom.findall("m:profiles", namespace))
            if count > 1:
                duplicates.append(f"{relative_path} ({count})")
        self.assertEqual([], duplicates)

    def test_shared_variant_names_preserve_workflow_matrix_semantics(self):
        self.assertEqual("mps-compile", build_platform.shared_variant_helper({"name": "mps-compile", "helper": "mps", "mlir": True}))
        self.assertEqual("compile-nnapi", build_platform.shared_variant_helper({"name": "compile-nnapi", "mlir": True}))
        self.assertEqual("compile", build_platform.shared_variant_helper({"name": "compile", "mlir": True}))
        self.assertEqual("compile", build_platform.shared_variant_helper({"name": "compile", "triton": True}))

    def test_zluda_release_contract_is_explicit_for_every_cloud_plan(self):
        root = Path(__file__).parents[2]
        expected = {
            "linux-x86_64-zluda": {
                "os": "linux",
                "platform": "linux-x86_64",
                "profiles": ["cuda", "sdx", "zluda", "zluda-platform"],
                "modules": {
                    ":nd4j-cuda-backend-common",
                    ":nd4j-cuda-12.9-preset",
                    ":nd4j-zluda-12.9",
                    ":nd4j-zluda-12.9-platform",
                    ":nd4j-presets-common",
                    ":libnd4j",
                },
                "artifactIds": {
                    "nd4j-cuda-backend-common",
                    "nd4j-cuda-12.9-preset",
                    "nd4j-zluda-12.9",
                    "nd4j-zluda-12.9-platform",
                    "nd4j-presets-common",
                },
                "unclassifiedArtifactIds": [
                    "nd4j-cuda-backend-common",
                    "nd4j-cuda-12.9-preset",
                    "nd4j-zluda-12.9",
                    "nd4j-zluda-12.9-platform",
                    "nd4j-presets-common",
                ],
            },
            "windows-x86_64-zluda": {
                "os": "windows",
                "platform": "windows-x86_64",
                "profiles": ["cuda", "sdx", "zluda", "zluda-platform"],
                "modules": {
                    ":nd4j-cuda-backend-common",
                    ":nd4j-cuda-12.9-preset",
                    ":nd4j-zluda-12.9",
                    ":nd4j-zluda-12.9-platform",
                    ":nd4j-presets-common",
                    ":libnd4j",
                },
                "artifactIds": {
                    "nd4j-cuda-backend-common",
                    "nd4j-cuda-12.9-preset",
                    "nd4j-zluda-12.9",
                    "nd4j-zluda-12.9-platform",
                    "nd4j-presets-common",
                },
                "unclassifiedArtifactIds": [
                    "nd4j-cuda-backend-common",
                    "nd4j-cuda-12.9-preset",
                    "nd4j-zluda-12.9",
                    "nd4j-zluda-12.9-platform",
                    "nd4j-presets-common",
                ],
            },
        }
        for provider in ("aws", "gcp", "azure"):
            plan = json.loads((root / f"release/{provider}/release-plan.json").read_text(encoding="utf-8"))
            for shard_id, expectation in expected.items():
                with self.subTest(provider=provider, shard=shard_id):
                    shard = next(item for item in plan["shards"] if item["id"] == shard_id)
                    self.assertEqual(expectation["os"], shard["os"])
                    build = shard["build"]
                    rules = shard["artifactRules"]
                    platform = expectation["platform"]
                    self.assertEqual(platform, build["javacppPlatform"])
                    self.assertEqual("12.9", build["cudaVersion"])
                    self.assertEqual("v7-preview.8", build["zludaVersion"])
                    if expectation["os"] == "linux":
                        self.assertEqual("7.2.4", build["rocmVersion"])
                        self.assertIs(True, build["rocmBuildOnly"])
                        self.assertEqual(
                            [
                                "hip", "rocblas", "hipblaslt", "rocsparse",
                                "rocm-smi", "miopen",
                            ],
                            build["rocmBuildComponents"],
                        )
                        archive_contract = rules["classifierArchiveContracts"][
                            "nd4j-zluda-12.9"
                        ]
                        self.assertEqual(
                            {
                                "libcuda.so": "libnvcuda.so",
                                "libcuda.so.1": "libnvcuda.so",
                            },
                            archive_contract["requiredRuntimeAliases"],
                        )
                    else:
                        self.assertEqual("7.2.4", build["rocmVersion"])
                        self.assertNotIn("rocmBuildOnly", build)
                        self.assertNotIn("rocmBuildComponents", build)
                    self.assertEqual(expectation["profiles"], build["profiles"])
                    self.assertEqual(expectation["modules"], set(build["modules"]))
                    self.assertEqual([{
                        "name": "cuda-12.9",
                        "classifierSuffix": "-cuda-12.9-zluda-rocm-7.2.4",
                        "platformExtension": "-zluda-rocm-7.2.4",
                    }], build["variants"])
                    self.assertIn("-Dlibnd4j.zluda=AMD", build["mavenArgs"])
                    self.assertIn("-Drocm.version=7.2.4", build["mavenArgs"])
                    self.assertNotIn("-Dlibnd4j.zluda=rocm6", build["mavenArgs"])
                    self.assertEqual(expectation["artifactIds"], set(rules["artifactIds"]))
                    self.assertEqual("nd4j-zluda-12.9", rules.get("classifierPrimaryArtifact"))
                    self.assertEqual(
                        [f"{platform}-zluda-rocm-7.2.4"],
                        shard["artifactRules"]["classifierTokens"],
                    )
                    self.assertEqual(
                        f"{platform}-zluda-rocm-7.2.4",
                        build_platform.variant_artifact_classifier(
                            build, build["variants"][0]
                        ),
                    )
                    self.assertEqual(
                        expectation["unclassifiedArtifactIds"],
                        rules.get("unclassifiedArtifactIds", []),
                    )

    def test_rocm_6_2_4_zluda_shards_are_published_for_every_cloud(self):
        root = Path(__file__).parents[2]
        expected = {
            "linux-x86_64-zluda-rocm-6.2.4": ("linux", "linux-x86_64"),
            "windows-x86_64-zluda-rocm-6.2.4": ("windows", "windows-x86_64"),
        }
        for provider in ("aws", "gcp", "azure"):
            plan = json.loads(
                (root / f"release/{provider}/release-plan.json").read_text(encoding="utf-8")
            )
            by_id = {shard["id"]: shard for shard in plan["shards"]}
            for shard_id, (os_name, platform_name) in expected.items():
                with self.subTest(provider=provider, shard=shard_id):
                    shard = by_id[shard_id]
                    build = shard["build"]
                    self.assertEqual(os_name, shard["os"])
                    self.assertEqual(platform_name, build["javacppPlatform"])
                    self.assertEqual("6.2.4", build["rocmVersion"])
                    self.assertEqual(
                        [f"{platform_name}-zluda-rocm-6.2.4"],
                        shard["artifactRules"]["classifierTokens"],
                    )
                    self.assertEqual(
                        "-cuda-12.9-zluda-rocm-6.2.4",
                        build["variants"][0]["classifierSuffix"],
                    )
                    self.assertEqual(
                        "-zluda-rocm-6.2.4",
                        build["variants"][0]["platformExtension"],
                    )
                    self.assertIn("-Drocm.version=6.2.4", build["mavenArgs"])
                    if os_name == "linux":
                        self.assertTrue(build["rocmBuildOnly"])
                        self.assertEqual(
                            [
                                "hip", "rocblas", "hipblaslt", "rocsparse",
                                "rocm-smi", "miopen",
                            ],
                            build["rocmBuildComponents"],
                        )
                    else:
                        self.assertNotIn("rocmBuildOnly", build)
                        self.assertNotIn("rocmBuildComponents", build)

    def test_rocm_hsakmt_sources_match_each_upstream_archive_layout(self):
        standalone = build_platform.ROCM_BUILD_SDKS["6.2.4"]
        self.assertEqual(
            "https://codeload.github.com/ROCm/ROCT-Thunk-Interface/tar.gz/"
            "7f307277e71e695dae11e600182a3f5bb53b95b4",
            standalone["hsakmt_source_url"],
        )
        self.assertEqual("", standalone["hsakmt_source_subdirectory"])
        self.assertEqual("", standalone["hsakmt_cmake_subdirectory"])
        self.assertFalse(standalone["hsakmt_rewrite_static_target"])
        self.assertTrue(standalone["hsakmt_disable_static_drm_target"])

        monorepo = build_platform.ROCM_BUILD_SDKS["7.2.4"]
        self.assertEqual("projects/rocr-runtime", monorepo["hsakmt_source_subdirectory"])
        self.assertEqual("libhsakmt", monorepo["hsakmt_cmake_subdirectory"])
        self.assertTrue(monorepo["hsakmt_rewrite_static_target"])
        self.assertFalse(monorepo["hsakmt_disable_static_drm_target"])

        with tempfile.TemporaryDirectory() as temp:
            extracted = Path(temp)
            standalone_root = extracted / "ROCT-Thunk-Interface-rocm-6.2.4"
            standalone_root.mkdir()
            (standalone_root / "CMakeLists.txt").write_text(
                "project(hsakmt)\n", encoding="utf-8"
            )
            candidates = build_platform.rocm_hsakmt_source_candidates(
                extracted, standalone
            )
            self.assertEqual([standalone_root / "CMakeLists.txt"], candidates)

        with tempfile.TemporaryDirectory() as temp:
            extracted = Path(temp)
            monorepo_root = (
                extracted / "rocm-systems-rocm-7.2.4" / "projects" / "rocr-runtime"
            )
            monorepo_root.mkdir(parents=True)
            (monorepo_root / "CMakeLists.txt").write_text(
                "project(rocr-runtime)\n", encoding="utf-8"
            )
            candidates = build_platform.rocm_hsakmt_source_candidates(
                extracted, monorepo
            )
            self.assertEqual([monorepo_root / "CMakeLists.txt"], candidates)

    def test_rocm_hsakmt_static_target_is_guarded_for_shared_build(self):
        source = (
            "## Create separate target file for static builds\n"
            "add_library ( ${HSAKMT_STATIC_DRM_TARGET} STATIC ${HSAKMT_SRC})\n"
            "install ( EXPORT ${HSAKMT_STATIC_DRM_TARGET}Targets\n"
            "  COMPONENT devel)\n"
            "\n###########################\n"
            "# Packaging directives\n"
        )
        adapted = build_platform.disable_rocm_hsakmt_static_target(source)
        self.assertIn(
            "if ( NOT BUILD_SHARED_LIBS)\n## Create separate target file for static builds",
            adapted,
        )
        self.assertIn("\nendif()\n\n###########################", adapted)
        self.assertEqual(
            adapted,
            build_platform.disable_rocm_hsakmt_static_target(adapted),
        )
        with self.assertRaisesRegex(RuntimeError, "static HSAKMT target block"):
            build_platform.disable_rocm_hsakmt_static_target("# Packaging directives\n")

    def test_windows_workers_import_visual_studio_environment_before_build(self):
        root = Path(__file__).parents[2]
        for provider in ("aws", "gcp", "azure"):
            with self.subTest(provider=provider):
                worker = (root / f"release/{provider}/worker.ps1").read_text(
                    encoding="utf-8"
                )
                self.assertEqual(2, worker.count("Import-VisualStudioEnvironment"))
                self.assertIn(
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    worker,
                )
                self.assertIn("-version '[17.0,18.0)'", worker)
                self.assertIn(r"VC\Auxiliary\Build\vcvars64.bat", worker)
                self.assertIn(
                    '$env:ComSpec /d /s /c "`"$VcVars`" >nul && set"',
                    worker,
                )
                self.assertIn(
                    "[Environment]::SetEnvironmentVariable("
                    "$Name, $Value, 'Process')",
                    worker,
                )
                self.assertIn("$env:VCINSTALLDIR", worker)
                self.assertIn("Get-Command cl.exe", worker)
                self.assertGreater(
                    worker.rindex("Import-VisualStudioEnvironment"),
                    worker.index("visualstudio2022-workload-vctools"),
                )
                self.assertLess(
                    worker.rindex("Import-VisualStudioEnvironment"),
                    worker.index("$Arguments ="),
                )

    def test_zluda_cmake_runtime_contracts_are_registered(self):
        root = Path(__file__).parents[2]
        cmake_source = (root / "libnd4j/CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("NAME zluda_windows_runtime_contract", cmake_source)
        self.assertIn("cmake/tests/ZludaWindowsRuntimeContractTest.cmake", cmake_source)
        self.assertIn("NAME shared_runtime_alias_contract", cmake_source)
        self.assertIn("cmake/tests/SharedRuntimeAliasContractTest.cmake", cmake_source)

    def test_zluda_consumer_contract_is_self_contained_and_amd_only(self):
        root = Path(__file__).parents[2]
        cuda_configuration = (
            root / "libnd4j/cmake/CudaConfiguration.cmake"
        ).read_text(encoding="utf-8")
        sdx_cuda_configuration = (
            root / "libnd4j/cmake/BuildSDX.cmake"
        ).read_text(encoding="utf-8")
        runtime_staging = (
            root / "libnd4j/cmake/StageSharedRuntime.cmake"
        ).read_text(encoding="utf-8")
        zluda_configuration = (
            root / "libnd4j/cmake/ZludaConfiguration.cmake"
        ).read_text(encoding="utf-8")
        dependency_configuration = (
            root / "libnd4j/cmake/Dependencies.cmake"
        ).read_text(encoding="utf-8")
        zluda_options = (root / "libnd4j/cmake/Options.cmake").read_text(
            encoding="utf-8"
        )
        backend = (
            root
            / "nd4j/nd4j-backends/nd4j-backend-impls/nd4j-zluda"
            / "src/main/java/org/nd4j/linalg/jzluda/JZludaBackend.java"
        ).read_text(encoding="utf-8")
        environment = (
            root
            / "nd4j/nd4j-backends/nd4j-backend-impls/nd4j-zluda"
            / "src/main/java/org/nd4j/linalg/jzluda/ZludaEnvironment.java"
        ).read_text(encoding="utf-8")
        dsp_runtime = (
            root / "libnd4j/include/legacy/impl/DspRuntimeC.cpp"
        ).read_text(encoding="utf-8")

        self.assertIn("CUDA::cudart", cuda_configuration)
        self.assertIn("CUDA::nvrtc", cuda_configuration)
        self.assertIn("link_zluda_cuda_shared_library", cuda_configuration)
        self.assertIn('target_link_options(${main_target_name} PRIVATE "LINKER:--no-as-needed")', cuda_configuration)
        self.assertNotIn("CUDA::cudart_static", cuda_configuration)
        self.assertNotIn("CUDA::nvrtc_static", cuda_configuration)
        self.assertIn('TARGET CUDA::cudart', sdx_cuda_configuration)
        self.assertIn('"$<TARGET_FILE:CUDA::cudart>"', sdx_cuda_configuration)
        self.assertIn('TARGET CUDA::nvrtc', sdx_cuda_configuration)
        self.assertIn('"$<TARGET_FILE:CUDA::nvrtc>"', sdx_cuda_configuration)
        self.assertNotIn("CUDA::cusolver_static", cuda_configuration)
        self.assertNotIn("cusolver_lapack_static", cuda_configuration)
        self.assertIn('RUNTIME_POLICY=$<IF:$<BOOL:${SD_ZLUDA}>,zluda-amd,default>', cuda_configuration)
        self.assertIn("ZLUDA backend requires", runtime_staging)
        self.assertIn("ZLUDA_RUNTIME_LIBRARIES", cuda_configuration)
        self.assertIn("ROCM_HIP_RUNTIME_LIBRARY", cuda_configuration)
        self.assertIn("setup_zluda_download(ZLUDA_MANAGED_ROOT)", zluda_configuration)
        self.assertIn("The published ZLUDA backend is AMD-only", zluda_configuration)
        self.assertIn("function(setup_zluda_download output_root)", dependency_configuration)
        self.assertIn("EXPECTED_HASH \"SHA256=${_zluda_sha256}\"", dependency_configuration)
        self.assertIn("zluda-linux-3fe12063.tar.gz", dependency_configuration)
        self.assertIn("zluda-windows-3fe1206.zip", dependency_configuration)
        self.assertNotIn("ZLUDA_ROOT", zluda_configuration)
        self.assertNotIn("setup_zluda_intel", zluda_configuration)
        self.assertNotIn("ONEAPI", zluda_configuration)
        self.assertNotIn("ENV{ZLUDA_PATH}", dependency_configuration)
        self.assertIn(
            'set_property(CACHE SD_ZLUDA_TARGET PROPERTY STRINGS AMD)',
            zluda_options,
        )

        self.assertIn("Loader.load(Nd4jCuda.class)", backend)
        self.assertNotIn('System.getenv("ZLUDA_PATH")', backend)
        self.assertNotIn('System.getenv("ZLUDA_TARGET")', backend)
        self.assertNotIn("INTEL", backend)
        self.assertNotIn("ONEAPI_ROOT", environment)
        self.assertNotIn('setenv("ZLUDA_TARGET"', dsp_runtime)
        self.assertNotIn('_putenv_s("ZLUDA_TARGET"', dsp_runtime)

        namespace = {"m": "http://maven.apache.org/POM/4.0.0"}
        zluda_pom = ET.parse(
            root
            / "nd4j/nd4j-backends/nd4j-backend-impls/nd4j-zluda/pom.xml"
        ).getroot()
        runtime_dependencies = {
            item.findtext("m:artifactId", namespaces=namespace)
            for item in zluda_pom.findall("m:dependencies/m:dependency", namespace)
        }
        self.assertNotIn("nd4j-cuda-${cuda.version}", runtime_dependencies)
        self.assertIn("nd4j-cuda-backend-common", runtime_dependencies)
        profile = next(
            item
            for item in zluda_pom.findall("m:profiles/m:profile", namespace)
            if item.findtext("m:id", namespaces=namespace) == "zluda-native"
        )
        profile_artifacts = {
            item.findtext("m:artifactId", namespaces=namespace)
            for item in profile.findall("m:dependencies/m:dependency", namespace)
        }
        self.assertNotIn("nd4j-cuda-${cuda.version}", profile_artifacts)
        self.assertIn("libnd4j", profile_artifacts)
        self.assertIn("nd4j-cuda-${cuda.version}-preset", profile_artifacts)
        self.assertIsNone(profile.find(".//m:artifactId[.='maven-dependency-plugin']", namespace))
        attached_classifier = profile.find(
            ".//m:execution[m:id='attach-zluda-native-classifier']"
            "/m:configuration/m:classifier",
            namespace,
        )
        self.assertIsNotNone(attached_classifier)
        self.assertEqual(
            "${javacpp.platform}${javacpp.platform.extension}",
            attached_classifier.text,
        )
        classifier_excludes = profile.find(
            ".//m:execution[m:id='attach-zluda-native-classifier']"
            "/m:configuration/m:excludes",
            namespace,
        )
        self.assertIsNotNone(classifier_excludes)
        self.assertEqual("override", classifier_excludes.attrib.get("combine.self"))
        self.assertEqual([], list(classifier_excludes))
        direct_jni_build = profile.find(
            ".//m:execution[m:id='build-zluda-jni']", namespace
        )
        self.assertIsNotNone(direct_jni_build)
        self.assertEqual(
            "org.nd4j.linalg.jcublas.bindings.Nd4jCuda",
            direct_jni_build.findtext(
                "m:configuration/m:classOrPackageName", namespaces=namespace
            ),
        )
        cleanup_execution = profile.find(
            ".//m:execution[m:id='cleanup-zluda-cmake-build']", namespace
        )
        self.assertIsNotNone(cleanup_execution)
        self.assertEqual(
            "prepare-package", cleanup_execution.findtext("m:phase", namespaces=namespace)
        )
        cleanup_fileset = cleanup_execution.find(
            ".//m:fileset", namespace
        )
        self.assertIsNotNone(cleanup_fileset)
        excludes = cleanup_fileset.attrib.get("excludes", "")
        self.assertIn("zluda-runtime-package/**", excludes)
        self.assertIn("javacpp-build-toolchain.properties", excludes)

    def test_zluda_cmake_scopes_rocm_to_amd_native_call_sites(self):
        root = Path(__file__).parents[2]
        configuration = (
            root / "libnd4j/cmake/ZludaConfiguration.cmake"
        ).read_text(encoding="utf-8")
        replay_factory = (
            root / "libnd4j/include/graph/impl/GraphReplayFactory.cpp"
        ).read_text(encoding="utf-8")
        gpu_backend = (
            root / "libnd4j/include/graph/impl/NativeDynamicShapePlan_gpubackend.cpp"
        ).read_text(encoding="utf-8")
        backend_catalog = (
            root / "libnd4j/include/graph/impl/NativeDynamicShapePlan_segments.cpp"
        ).read_text(encoding="utf-8")
        miopen_bridge = (
            root / "libnd4j/include/ops/declarable/platform/miopen/miopenBridge.cpp"
        ).read_text(encoding="utf-8")
        miopen_cuda_call_sites = "\n".join(
            (root / path).read_text(encoding="utf-8")
            for path in (
                "libnd4j/include/ops/declarable/platform/miopen/miopenBridge.h",
                "libnd4j/include/ops/declarable/platform/miopen/miopenUtils.h",
                "libnd4j/include/ops/declarable/platform/miopen/activations.cpp",
                "libnd4j/include/ops/declarable/platform/miopen/batchnorm.cpp",
                "libnd4j/include/ops/declarable/platform/miopen/conv2d.cpp",
                "libnd4j/include/ops/declarable/platform/miopen/softmax.cpp",
            )
        )
        direct_hip_sources = "\n".join(
            (root / path).read_text(encoding="utf-8")
            for path in (
                "libnd4j/include/graph/hip/HipGraphBackend.cpp",
                "libnd4j/include/graph/hip/HipGraphBackend.h",
                "libnd4j/include/graph/hip/HipRuntimeManager.cpp",
                "libnd4j/include/graph/hip/HipRuntimeManager.h",
                "libnd4j/include/graph/hip/HipGraphReplayHandle.cpp",
                "libnd4j/include/graph/hip/HipGraphReplayHandle.h",
            )
        )

        for token in (
            "ROCM_HIP_RUNTIME_LIBRARY",
            "DL4J_ZLUDA_REQUIRE_ROCM",
            "DL4J_ZLUDA_REQUIRE_MIOPEN",
            "target_link_libraries(${target_name} PRIVATE ${MIOPEN_LIBRARY})",
            "target_link_libraries(${target_name} PRIVATE ${ROCM_HIP_RUNTIME_LIBRARY})",
        ):
            self.assertIn(token, configuration)
        for leaked_sdk_setting in (
            'include_directories(SYSTEM "${ROCM_INCLUDE_DIR}")',
            "target_include_directories(${target_name} SYSTEM PUBLIC ${ROCM_INCLUDE_DIR})",
            "target_link_libraries(${target_name} PUBLIC ${ROCM_HIP_RUNTIME_LIBRARY})",
            "add_compile_definitions(__HIP_PLATFORM_AMD__=1)",
            'COMPILE_DEFINITIONS "__HIP_PLATFORM_NVIDIA__=1"',
        ):
            self.assertNotIn(leaked_sdk_setting, configuration)

        self.assertIn(
            'COMPILE_DEFINITIONS "__HIP_PLATFORM_AMD__=1"', configuration
        )
        self.assertIn(
            'set(_ZLUDA_MIOPEN_BRIDGE_SOURCE', configuration
        )
        self.assertIn(
            'INCLUDE_DIRECTORIES "${_ZLUDA_MIOPEN_INCLUDE_DIRS}"', configuration
        )
        self.assertIn("SKIP_UNITY_BUILD_INCLUSION ON", configuration)
        self.assertIn("#include <hip/hip_runtime.h>", miopen_bridge)
        self.assertIn("#include <miopen/miopen.h>", miopen_bridge)
        self.assertNotIn("#include <hip/", miopen_cuda_call_sites)
        self.assertNotIn("#include <miopen/", miopen_cuda_call_sites)
        old_hip_gate = (
            "#if defined(SD_HIP) || defined(ZLUDA_TARGET_AMD) || "
            "defined(HAVE_MIOPEN)"
        )
        self.assertNotIn(old_hip_gate, replay_factory)
        self.assertNotIn(old_hip_gate, gpu_backend)
        self.assertNotIn(old_hip_gate, backend_catalog)
        self.assertNotIn(old_hip_gate, direct_hip_sources)
        self.assertIn("#if defined(SD_HIP)", replay_factory)
        self.assertIn("#if defined(SD_HIP)", backend_catalog)
        self.assertEqual(6, direct_hip_sources.count("#if defined(SD_HIP)"))

        setup_call = configuration.index("setup_zluda_amd()")
        propagation = configuration.index(
            "ROCM_PATH ROCM_INCLUDE_DIR ROCM_LIB_DIR ROCM_HIP_RUNTIME_LIBRARY",
            setup_call,
        )
        self.assertLess(setup_call, propagation)
        for poisoned_lookup in (
                'set(ROCM_HIP_RUNTIME_LIBRARY "")',
                'set(MIOPEN_LIBRARY "")',
                'set(MIOPEN_INCLUDE_DIR "")'):
            self.assertNotIn(poisoned_lookup, configuration)

    def test_linux_zluda_worker_installs_only_build_time_rocm_components(self):
        root = Path(__file__).parents[2]
        plan = json.loads(
            (root / "release/aws/release-plan.json").read_text(encoding="utf-8")
        )
        build = next(
            shard["build"] for shard in plan["shards"]
            if shard["id"] == "linux-x86_64-zluda"
        )
        driver = (root / "release/aws/build-platform.py").read_text(encoding="utf-8")
        self.assertEqual("7.2.4", build["rocmVersion"])
        self.assertTrue(build["rocmBuildOnly"])
        self.assertEqual(
            ["hip", "rocblas", "hipblaslt", "rocsparse", "rocm-smi", "miopen"],
            build["rocmBuildComponents"],
        )
        for package in (
            "lld", "patchelf", "rocm-hip-runtime-dev", "rocblas-dev",
            "hipblaslt-dev", "rocsparse-dev", "rocm-smi-lib", "miopen-hip-dev",
        ):
            self.assertIn(package, driver)
        self.assertNotIn("ZLUDA_PATH", driver)
        self.assertIn('env["DL4J_ZLUDA_REQUIRE_ROCM"] = "1"', driver)
        self.assertIn('env["DL4J_ZLUDA_REQUIRE_MIOPEN"] = "1"', driver)
        self.assertIn("hardwareProbe=skipped", driver)
        self.assertNotIn("amdgpu-dkms", driver)
        self.assertNotIn("rocminfo", driver)

    def test_linux_zluda_uses_large_code_model_for_multi_gibibyte_library(self):
        root = Path(__file__).parents[2]
        configuration = (
            root / "libnd4j/cmake/CudaConfiguration.cmake"
        ).read_text(encoding="utf-8")
        compiler_flags = (
            root / "libnd4j/cmake/CompilerFlags.cmake"
        ).read_text(encoding="utf-8")
        large_binary_start = configuration.index(
            "if(SD_GCC_FUNCTRACE OR SD_ZLUDA)"
        )
        linker_selection = configuration.index(
            "# Linker selection for large binaries", large_binary_start
        )
        large_binary_flags = configuration[large_binary_start:linker_selection]

        self.assertIn(
            'if(SD_ZLUDA)\n'
            '                    set(DL4J_LARGE_BINARY_CODE_MODEL "large")\n'
            '                else()\n'
            '                    set(DL4J_LARGE_BINARY_CODE_MODEL "medium")\n'
            '                endif()',
            large_binary_flags,
        )
        self.assertIn(
            "-Xcompiler=-mcmodel=${DL4J_LARGE_BINARY_CODE_MODEL}",
            large_binary_flags,
        )
        self.assertIn(
            'if(SD_ZLUDA)\n'
            '                    # GCC 11 emits 32-bit jump-table references'
            ' at the project\'s\n'
            '                    # -O level even under the large code model.',
            large_binary_flags,
        )
        self.assertIn(
            "-Xcompiler=-fno-jump-tables",
            large_binary_flags,
        )
        self.assertIn(
            'if(SD_ZLUDA)\n'
            '                    set(CMAKE_CXX_FLAGS '
            '"${CMAKE_CXX_FLAGS} -fno-jump-tables")',
            configuration,
        )
        self.assertIn(
            "-mcmodel=${DL4J_LARGE_BINARY_CODE_MODEL}", configuration
        )

        memory_model_start = compiler_flags.index(
            "# --- Memory Model for large binaries ---"
        )
        section_splitting = compiler_flags.index(
            "# --- Section splitting for better linker handling ---",
            memory_model_start,
        )
        memory_model = compiler_flags[memory_model_start:section_splitting]
        self.assertIn(
            'if(SD_ZLUDA)\n'
            '        # GCC 11 can still emit R_X86_64_PC32 references from switch code to\n'
            '        # jump tables at -O even with -mcmodel=large. The all-ops ZLUDA image\n'
            '        # can place those tables more than 2 GiB away, so use comparison trees\n'
            '        # for this classifier\'s host code instead.\n'
            '        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcmodel=large -fPIC -fno-jump-tables")\n'
            '        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mcmodel=large -fno-jump-tables")',
            memory_model,
        )
        self.assertIn("elseif(SD_SANITIZE OR SD_GCC_FUNCTRACE)", memory_model)
        self.assertIn("-mcmodel=medium", memory_model)

        strict_linker_start = compiler_flags.index(
            "# --- Strict Linker Flags to Catch Undefined Symbols Early ---"
        )
        functrace_linker = compiler_flags.index(
            "elseif(CMAKE_SYSTEM_NAME STREQUAL \"Linux\" AND SD_GCC_FUNCTRACE",
            strict_linker_start,
        )
        strict_linker = compiler_flags[strict_linker_start:functrace_linker]
        self.assertIn(
            'if(SD_ZLUDA)\n'
            '            set(CMAKE_SHARED_LINKER_FLAGS '
            '"${CMAKE_SHARED_LINKER_FLAGS} -mcmodel=large")',
            strict_linker,
        )

    def test_linux_zluda_requires_lld_with_section_gc(self):
        root = Path(__file__).parents[2]
        configuration = (
            root / "libnd4j/cmake/CudaConfiguration.cmake"
        ).read_text(encoding="utf-8")
        zluda_branch = configuration.index("if(SD_ZLUDA)")
        lld_selection = configuration.index(
            'set(LINKER_FLAG "-fuse-ld=lld")', zluda_branch
        )
        gold_fallback = configuration.index("elseif(GOLD_LINKER)", zluda_branch)
        self.assertLess(lld_selection, gold_fallback)
        self.assertIn("if(NOT LLD_LINKER)", configuration[zluda_branch:gold_fallback])
        self.assertIn(
            "-Wl,--icf=all -Wl,--gc-sections -Wl,--as-needed -Wl,-z,notext",
            configuration[zluda_branch:gold_fallback],
        )

    def test_openblas_path_is_normalized_before_config_header_generation(self):
        root = Path(__file__).parents[2]
        dependencies = (
            root / "libnd4j/cmake/Dependencies.cmake"
        ).read_text(encoding="utf-8")
        normalization = 'string(REPLACE "\\\\" "/" OPENBLAS_PATH "${OPENBLAS_PATH}")'
        propagation = 'set(OPENBLAS_PATH "${OPENBLAS_PATH}" PARENT_SCOPE)'
        validation = 'if(NOT EXISTS "${OPENBLAS_PATH}/include")'
        self.assertIn(normalization, dependencies)
        self.assertIn(propagation, dependencies)
        self.assertLess(dependencies.index(normalization), dependencies.index(propagation))
        self.assertLess(dependencies.index(propagation), dependencies.index(validation))

    def test_onednn_cache_restore_uses_cold_build_platform_libdir(self):
        root = Path(__file__).parents[2]
        dependencies = (
            root / "libnd4j/cmake/Dependencies.cmake"
        ).read_text(encoding="utf-8")
        setup_start = dependencies.index("function(setup_onednn)")
        setup_end = dependencies.index("endfunction()", setup_start)
        setup = dependencies[setup_start:setup_end]
        libdir_selection = (
            'if(WIN32)\n'
            '        set(ONEDNN_LIB_DIR "lib")\n'
            '    else()\n'
            '        set(ONEDNN_LIB_DIR "lib64")\n'
            '    endif()'
        )
        cache_start = setup.index("if(_onednn_hit)")
        cache_end = setup.index("return()", cache_start)
        cache_hit = setup[cache_start:cache_end]

        self.assertIn(libdir_selection, setup)
        self.assertLess(
            setup.index(libdir_selection),
            setup.index("# --- Dependency cache check ---"),
        )
        self.assertIn(
            '"${ONEDNN_INSTALL_DIR}/${ONEDNN_LIB_DIR}/dnnl.lib"',
            cache_hit,
        )
        self.assertIn(
            '"${ONEDNN_INSTALL_DIR}/${ONEDNN_LIB_DIR}/libdnnl.a"',
            cache_hit,
        )
        self.assertIn("-DCMAKE_INSTALL_LIBDIR=${ONEDNN_LIB_DIR}", setup)
        self.assertNotIn("${ONEDNN_INSTALL_DIR}/lib64/libdnnl.a", setup)

    def test_openvino_static_link_response_includes_all_installed_archives(self):
        project = Path(__file__).parents[2]
        generator = project / "libnd4j/cmake/install_openvino.cmake"
        dependencies = (
            project / "libnd4j/cmake/Dependencies.cmake"
        ).read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as temporary_directory:
            install = Path(temporary_directory) / "openvino_install"
            core = install / "runtime/lib/intel64"
            third_party = install / "runtime/3rdparty/lib"
            core.mkdir(parents=True)
            third_party.mkdir(parents=True)
            archives = [
                core / "libopenvino.a",
                core / "libopenvino_onednn_cpu.a",
                third_party / "libpugixml.a",
            ]
            for archive in archives:
                archive.write_bytes(b"!<arch>\n")
            response = core / "openvino-static-link.rsp"

            subprocess.run(
                [
                    "cmake",
                    f"-DINSTALL_DIR={install}",
                    f"-DRESPONSE_FILE={response}",
                    "-P",
                    str(generator),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            lines = response.read_text(encoding="utf-8").splitlines()
            self.assertEqual("--start-group", lines[0])
            self.assertEqual("--end-group", lines[len(archives) + 1])
            for archive in archives:
                self.assertIn(f'"{archive}"', lines)
            self.assertIn("-ltbb", lines)
            self.assertIn("-ltbbmalloc", lines)
            self.assertIn('"-Wl,@${_ov_link_response}"', dependencies)
            self.assertNotIn("libopenvino_onednn_cpu.a", dependencies)

    def test_managed_llvm_does_not_publish_private_zstd_dependency(self):
        root = Path(__file__).parents[2]
        dependencies = (
            root / "libnd4j/cmake/Dependencies.cmake"
        ).read_text(encoding="utf-8")
        setup_start = dependencies.index("function(setup_triton)")
        setup_end = dependencies.index("endfunction()", setup_start)
        setup = dependencies[setup_start:setup_end]

        self.assertNotIn("-lzstd", setup)
        self.assertEqual(
            2,
            setup.count(
                "target_link_libraries(triton_interface INTERFACE -lz -lm)"
            ),
        )

    def test_classifier_staging_keeps_only_explicit_unclassified_zluda_runtime(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            repository = root / "repository"
            output = root / "output"
            version = "1.0.0"
            artifacts = {
                "nd4j-zluda-12.9": [
                    f"nd4j-zluda-12.9-{version}.jar",
                    f"nd4j-zluda-12.9-{version}-sources.jar",
                    f"nd4j-zluda-12.9-{version}-wrong-classifier.jar",
                ],
                "nd4j-zluda-12.9-platform": [
                    f"nd4j-zluda-12.9-platform-{version}.jar",
                    f"nd4j-zluda-12.9-platform-{version}-sources.jar",
                ],
                "nd4j-cuda-12.9": [
                    f"nd4j-cuda-12.9-{version}.jar",
                    f"nd4j-cuda-12.9-{version}-linux-x86_64-zluda-rocm-7.2.4.jar",
                ],
            }
            for artifact_id, names in artifacts.items():
                directory = repository / "org/nd4j" / artifact_id / version
                directory.mkdir(parents=True)
                (directory / f"{artifact_id}-{version}.pom").write_text(
                    "<project/>", encoding="utf-8"
                )
                for name in names:
                    (directory / name).write_bytes(b"jar")

            build_platform.stage_repository(repository, output, {
                "mode": "classifier",
                "artifactIds": list(artifacts),
                "classifierTokens": ["linux-x86_64-zluda-rocm-7.2.4"],
                "unclassifiedArtifactIds": [
                    "nd4j-zluda-12.9",
                    "nd4j-zluda-12.9-platform",
                ],
                "includeMetadata": False,
            })

            staged = {path.name for path in output.rglob("*.jar")}
            staged_poms = {path.name for path in output.rglob("*.pom")}
            self.assertEqual(
                {f"{artifact_id}-{version}.pom" for artifact_id in artifacts},
                staged_poms,
            )
            self.assertIn(f"nd4j-zluda-12.9-{version}.jar", staged)
            self.assertIn(f"nd4j-zluda-12.9-platform-{version}.jar", staged)
            self.assertIn(f"nd4j-cuda-12.9-{version}-linux-x86_64-zluda-rocm-7.2.4.jar", staged)
            self.assertNotIn(f"nd4j-cuda-12.9-{version}.jar", staged)
            self.assertNotIn(f"nd4j-zluda-12.9-{version}-sources.jar", staged)
            self.assertNotIn(f"nd4j-zluda-12.9-platform-{version}-sources.jar", staged)
            self.assertNotIn(f"nd4j-zluda-12.9-{version}-wrong-classifier.jar", staged)
            with self.assertRaisesRegex(ValueError, "must be a subset"):
                build_platform.stage_repository(repository, output, {
                    "mode": "classifier",
                    "artifactIds": ["nd4j-cuda-12.9"],
                    "unclassifiedArtifactIds": ["nd4j-zluda-12.9"],
                })

    def test_zluda_unclassified_artifact_attestation_is_exact_and_complete(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = Path(temporary_directory)
            version = "1.0.0"
            build = {
                "modules": [":nd4j-zluda-12.9", ":nd4j-zluda-12.9-platform"],
            }
            rules = {
                "mode": "classifier",
                "artifactIds": ["nd4j-zluda-12.9", "nd4j-zluda-12.9-platform"],
                "unclassifiedArtifactIds": [
                    "nd4j-zluda-12.9",
                    "nd4j-zluda-12.9-platform",
                ],
            }
            zluda_jar = (
                repository
                / "org/nd4j/nd4j-zluda-12.9"
                / version
                / f"nd4j-zluda-12.9-{version}.jar"
            )
            platform_jar = (
                repository
                / "org/eclipse/deeplearning4j/nd4j-zluda-12.9-platform"
                / version
                / f"nd4j-zluda-12.9-platform-{version}.jar"
            )
            zluda_jar.parent.mkdir(parents=True)
            zluda_jar.write_bytes(b"jar")

            with self.assertRaisesRegex(RuntimeError, "nd4j-zluda-12.9-platform"):
                build_platform.attest_unclassified_artifacts(
                    repository,
                    build,
                    rules,
                    version,
                    "local-repository",
                )

            platform_jar.parent.mkdir(parents=True)
            platform_jar.write_bytes(b"jar")
            output = StringIO()
            with redirect_stdout(output):
                build_platform.attest_unclassified_artifacts(
                    repository,
                    build,
                    rules,
                    version,
                    "local-repository",
                )
            self.assertIn(
                "unclassified-artifacts="
                "org/eclipse/deeplearning4j/nd4j-zluda-12.9-platform/"
                f"{version}/nd4j-zluda-12.9-platform-{version}.jar,"
                f"org/nd4j/nd4j-zluda-12.9/{version}/"
                f"nd4j-zluda-12.9-{version}.jar",
                output.getvalue(),
            )

            build_platform.reset_unclassified_artifacts(
                repository,
                build,
                rules,
                version,
            )
            self.assertFalse(zluda_jar.exists())
            self.assertFalse(platform_jar.exists())

            with self.assertRaisesRegex(
                ValueError,
                "does not include required modules",
            ):
                build_platform.required_unclassified_artifact_ids(
                    {"modules": [":nd4j-zluda-12.9"]},
                    rules,
                )

    def test_cpu_cuda_classifier_artifact_attestation_is_exact_and_complete(self):
        cases = (
            (
                {
                    "backend": "cpu",
                    "javacppPlatform": "linux-x86_64",
                    "modules": [":nd4j-native", ":nd4j-native-preset", ":libnd4j"],
                },
                {
                    "name": "compile",
                    "suffix": "-compile",
                    "mlir": True,
                },
                ("nd4j-native", "nd4j-native-preset"),
            ),
            (
                {
                    "backend": "cuda",
                    "cudaVersion": "12.9",
                    "javacppPlatform": "linux-x86_64",
                    "modules": [
                        ":nd4j-cuda-12.9",
                        ":nd4j-cuda-12.9-preset",
                        ":libnd4j",
                    ],
                },
                {
                    "name": "compile",
                    "classifierSuffix": "-cuda-12.9-compile",
                    "platformExtension": "-compile",
                    "triton": True,
                },
                ("nd4j-cuda-12.9", "nd4j-cuda-12.9-preset"),
            ),
            (
                {
                    "backend": "cuda",
                    "cudaVersion": "12.9",
                    "zludaVersion": "v6",
                    "javacppPlatform": "linux-x86_64",
                    "modules": [
                        ":nd4j-cuda-12.9-preset",
                        ":nd4j-zluda-12.9",
                        ":nd4j-cuda-backend-common",
                        ":libnd4j",
                    ],
                },
                {
                    "name": "zluda",
                    "classifierSuffix": "-cuda-12.9-zluda-rocm-7.2.4",
                    "platformExtension": "-zluda-rocm-7.2.4",
                },
                ("nd4j-zluda-12.9", "nd4j-cuda-12.9-preset"),
            ),
        )
        version = "1.0.0-SNAPSHOT"
        for build, variant, artifact_ids in cases:
            with self.subTest(backend=build["backend"]), tempfile.TemporaryDirectory() as temp:
                repository = Path(temp)
                rules = {
                    "mode": "classifier",
                    "artifactIds": [*artifact_ids, f"{artifact_ids[0]}-platform"],
                }
                classifier = build_platform.variant_artifact_classifier(build, variant)

                def write_jar(artifact_id, artifact_classifier):
                    path = (
                        repository
                        / "org/eclipse/deeplearning4j"
                        / artifact_id
                        / version
                        / f"{artifact_id}-{version}-{artifact_classifier}.jar"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"jar")
                    return path

                lookalikes = [
                    write_jar(artifact_id, f"{classifier}-extra")
                    for artifact_id in artifact_ids
                ]
                with self.assertRaisesRegex(RuntimeError, "missing exact"):
                    build_platform.attest_variant_classifier_artifacts(
                        repository,
                        build,
                        rules,
                        variant,
                        version,
                        "test-repository",
                    )

                exact = [write_jar(artifact_ids[0], classifier)]
                with self.assertRaisesRegex(RuntimeError, artifact_ids[1]):
                    build_platform.attest_variant_classifier_artifacts(
                        repository,
                        build,
                        rules,
                        variant,
                        version,
                        "test-repository",
                    )
                exact.append(write_jar(artifact_ids[1], classifier))
                output = StringIO()
                with redirect_stdout(output):
                    build_platform.attest_variant_classifier_artifacts(
                        repository,
                        build,
                        rules,
                        variant,
                        version,
                        "test-repository",
                    )
                self.assertIn(f"classifier={classifier}", output.getvalue())
                self.assertTrue(all(path.is_file() for path in exact))

                build_platform.reset_variant_classifier_artifacts(
                    repository, build, rules, variant, version
                )
                self.assertTrue(all(not path.exists() for path in exact))
                self.assertTrue(all(path.is_file() for path in lookalikes))

    def test_zluda_classifier_archive_attestation_requires_complete_runtime_closure(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = Path(temporary_directory)
            version = "1.0.0-SNAPSHOT"
            classifier = "linux-x86_64-zluda-rocm-7.2.4"
            artifact_id = "nd4j-zluda-12.9"
            preset_id = "nd4j-cuda-12.9-preset"
            native_root = (
                "org/nd4j/linalg/jcublas/bindings/"
                f"{classifier}"
            )
            build = {
                "backend": "cuda",
                "cudaVersion": "12.9",
                "zludaVersion": "v6",
                "rocmVersion": "7.2.4",
                "javacppPlatform": "linux-x86_64",
                "modules": [f":{artifact_id}", f":{preset_id}"],
            }
            variant = {
                "name": "zluda",
                "classifierSuffix": "-cuda-12.9-zluda-rocm-7.2.4",
                "platformExtension": "-zluda-rocm-7.2.4",
            }
            rules = {
                "mode": "classifier",
                "artifactIds": [artifact_id, preset_id],
                "classifierArchiveContracts": {
                    artifact_id: {
                        "requiredEntries": [
                            f"{native_root}/libjnind4jcuda.so",
                            f"{native_root}/libnd4jcuda.so",
                            f"{native_root}/shared-runtime-manifest.txt",
                        ],
                        "runtimeManifest": (
                            f"{native_root}/shared-runtime-manifest.txt"
                        ),
                        "requiredRuntimeAliases": {
                            "libcuda.so": "libnvcuda.so",
                            "libcuda.so.1": "libnvcuda.so",
                        },
                    },
                },
            }

            def artifact_path(current_artifact_id):
                path = (
                    repository
                    / "org/eclipse/deeplearning4j"
                    / current_artifact_id
                    / version
                    / f"{current_artifact_id}-{version}-{classifier}.jar"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                return path

            preset_path = artifact_path(preset_id)
            preset_path.write_bytes(b"preset")
            zluda_path = artifact_path(artifact_id)
            manifest = (
                "# nd4j-shared-runtime-manifest-v1\n"
                "# runtime-count=2\n"
                "# runtime-alias-count=2\n"
                "# runtime-alias=libcuda.so->libnvcuda.so\n"
                "# runtime-alias=libcuda.so.1->libnvcuda.so\n"
                "libamdhip64.so\nlibnvcuda.so\n"
            )

            with zipfile.ZipFile(zluda_path, "w") as archive:
                archive.writestr(
                    f"{native_root}/shared-runtime-manifest.txt", manifest
                )
            with self.assertRaisesRegex(RuntimeError, "violates its runtime contract"):
                build_platform.attest_variant_classifier_artifacts(
                    repository, build, rules, variant, version, "test-repository"
                )

            with zipfile.ZipFile(zluda_path, "w") as archive:
                archive.writestr(f"{native_root}/libjnind4jcuda.so", b"jni")
                archive.writestr(f"{native_root}/libnd4jcuda.so", b"backend")
                archive.writestr(f"{native_root}/libnvcuda.so", b"zluda")
                archive.writestr(
                    f"{native_root}/shared-runtime-manifest.txt", manifest
                )
            with self.assertRaisesRegex(RuntimeError, "manifest-owned runtimes"):
                build_platform.attest_variant_classifier_artifacts(
                    repository, build, rules, variant, version, "test-repository"
                )

            with zipfile.ZipFile(zluda_path, "a") as archive:
                archive.writestr(f"{native_root}/libamdhip64.so", b"hip")
            with self.assertRaisesRegex(
                RuntimeError, "manifest-declared runtime aliases"
            ):
                build_platform.attest_variant_classifier_artifacts(
                    repository, build, rules, variant, version, "test-repository"
                )

            with zipfile.ZipFile(zluda_path, "a") as archive:
                archive.writestr(f"{native_root}/libcuda.so", b"zluda")
                archive.writestr(f"{native_root}/libcuda.so.1", b"zluda")
            output = StringIO()
            with redirect_stdout(output):
                build_platform.attest_variant_classifier_artifacts(
                    repository, build, rules, variant, version, "test-repository"
                )
            self.assertIn("runtime-closure=2", output.getvalue())
            self.assertIn("runtime-aliases=2", output.getvalue())

    def test_cpu_cuda_classifier_contract_rejects_incomplete_plan(self):
        build = {
            "backend": "cuda",
            "cudaVersion": "12.9",
            "modules": [":nd4j-cuda-12.9"],
        }
        with self.assertRaisesRegex(ValueError, "do not own required artifacts"):
            build_platform.required_classifier_artifact_ids(build, {
                "mode": "classifier",
                "artifactIds": ["nd4j-cuda-12.9"],
            })
        with self.assertRaisesRegex(ValueError, "does not include required modules"):
            build_platform.required_classifier_artifact_ids(build, {
                "mode": "classifier",
                "artifactIds": [
                    "nd4j-cuda-12.9",
                    "nd4j-cuda-12.9-preset",
                ],
            })

    def test_zluda_target_and_attestation_fail_closed(self):
        build = {
            "backend": "cuda",
            "cudaVersion": "12.9",
            "zludaVersion": "v6",
            "rocmVersion": "7.2.4",
            "javacppPlatform": "linux-x86_64",
            "profiles": ["cuda", "sdx", "zluda"],
            "modules": [":nd4j-cuda-12.9-preset", ":nd4j-zluda-12.9"],
            "mavenArgs": ["-Dlibnd4j.zluda=AMD"],
            "variants": [{
                "name": "zluda",
                "classifierSuffix": "-cuda-12.9-zluda-rocm-7.2.4",
                "platformExtension": "-zluda-rocm-7.2.4",
            }],
        }
        for platform in ("linux-x86_64", "windows-x86_64"):
            with self.subTest(platform=platform):
                output = StringIO()
                platform_build = dict(build, javacppPlatform=platform)
                with redirect_stdout(output):
                    build_platform.attest_zluda_configuration(platform_build)
                self.assertIn("target=AMD", output.getvalue())
                self.assertIn("dependencyOwner=cmake", output.getvalue())

        for arguments in ([], ["-Dlibnd4j.zluda=rocm6"], [
                "-Dlibnd4j.zluda=AMD", "-Dlibnd4j.zluda=AMD"]):
            with self.subTest(arguments=arguments):
                invalid = dict(build, mavenArgs=arguments)
                with self.assertRaises(ValueError):
                    build_platform.zluda_target(invalid)
        mediated = dict(build, modules=build["modules"] + [":nd4j-cuda-12.9"])
        with self.assertRaisesRegex(RuntimeError, "must not mediate"):
            build_platform.attest_zluda_configuration(mediated)

    def test_rocm_sdk_attestation_is_build_only_and_fail_closed(self):
        build = {
            "zludaVersion": "v6",
            "javacppPlatform": "linux-x86_64",
            "rocmVersion": "7.2.4",
            "rocmBuildOnly": True,
            "rocmBuildComponents": [
                "hip", "rocblas", "hipblaslt", "rocsparse", "rocm-smi", "miopen",
            ],
        }
        spec = build_platform.rocm_build_spec(build)
        self.assertEqual(
            (
                "rocm-hip-runtime-dev", "hsa-rocr-dev",
                "libnuma-dev", "libdrm-dev",
                "rocblas-dev", "hipblaslt-dev", "rocsparse-dev",
                "rocm-smi-lib", "miopen-hip-dev",
            ),
            spec["packages"],
        )
        six_spec = build_platform.rocm_build_spec({**build, "rocmVersion": "6.2.4"})
        self.assertIn("hsakmt-roct-dev", six_spec["packages"])
        self.assertTrue(six_spec["hsakmt_disable_static_drm_target"])
        self.assertFalse(spec["hsakmt_disable_static_drm_target"])
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            files = (
                root / ".info/version",
                root / "include/hip/hip_runtime.h",
                root / "bin/hipcc",
                root / "lib/libamdhip64.so",
                root / "include/rocblas/rocblas.h",
                root / "lib/librocblas.so",
                root / "lib/libhipblaslt.so",
                root / "lib/librocsparse.so",
                root / "lib/librocm_smi64.so",
                root / "include/miopen/miopen.h",
                root / "lib/libMIOpen.so",
                root / "lib/libhsa-runtime64.so.1",
                root / "lib/libhsakmt.so.1",
            )
            for path in files:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    "7.2.4.70204-93~22.04" if path == files[0] else "sdk",
                    encoding="utf-8",
                )
            environment = {"PATH": "/usr/bin"}
            output = StringIO()
            with redirect_stdout(output):
                attested = build_platform.attest_rocm_build_toolchain(
                    build, environment, root=root
                )
            self.assertEqual(files[1], attested["hipHeader"])
            self.assertEqual(files[11], attested["hsaRuntime"])
            self.assertEqual(files[12], attested["hsakmtRuntime"])
            self.assertEqual(str(root), environment["ROCM_PATH"])
            self.assertEqual("1", environment["DL4J_ZLUDA_REQUIRE_ROCM"])
            self.assertEqual("1", environment["DL4J_ZLUDA_REQUIRE_MIOPEN"])
            self.assertIn("hardwareProbe=skipped", output.getvalue())
            files[11].unlink()
            with self.assertRaisesRegex(RuntimeError, "HSA runtime"):
                build_platform.attest_rocm_build_toolchain(
                    build, environment, root=root
                )
            files[11].write_text("sdk", encoding="utf-8")
            files[12].unlink()
            with self.assertRaisesRegex(RuntimeError, "HSAKMT runtime"):
                build_platform.attest_rocm_build_toolchain(
                    build, environment, root=root
                )
            files[12].write_text("sdk", encoding="utf-8")
            files[9].unlink()
            with self.assertRaisesRegex(RuntimeError, "MIOpen header"):
                build_platform.attest_rocm_build_toolchain(
                    build, environment, root=root
                )

        with self.assertRaisesRegex(ValueError, "exact ROCm build components"):
            build_platform.rocm_build_spec(
                dict(build, rocmBuildComponents=["hip"])
            )
        with self.assertRaisesRegex(ValueError, "rocmBuildOnly=true"):
            build_platform.rocm_build_spec(dict(build, rocmBuildOnly=False))

    def test_rocm_sdk_provisioning_installs_no_kernel_driver(self):
        build = {
            "zludaVersion": "v6",
            "javacppPlatform": "linux-x86_64",
            "rocmVersion": "7.2.4",
            "rocmBuildOnly": True,
            "rocmBuildComponents": [
                "hip", "rocblas", "hipblaslt", "rocsparse", "rocm-smi", "miopen",
            ],
        }
        with patch.object(
                build_platform, "attest_rocm_build_toolchain",
                side_effect=[RuntimeError("missing"), None]) as attest, patch.object(
                build_platform, "download_with_retry") as download, patch.object(
                build_platform, "build_rocm_hsakmt", return_value=Path("/opt/rocm-7.2.4/lib/libhsakmt.so.1")) as hsakmt, patch.object(
                build_platform, "run") as run_command, patch.object(
                build_platform.platform, "system", return_value="Linux"), patch.object(
                build_platform.platform, "machine", return_value="x86_64"), patch.object(
                build_platform.os, "geteuid", return_value=0, create=True), patch.object(
                build_platform.shutil, "which",
                side_effect=[
                    None, None, "/usr/bin/ld.lld", "/usr/bin/patchelf",
                ]):
            environment = {}
            build_platform.prepare_rocm_build_toolchain(build, environment)
        download.assert_called_once()
        self.assertEqual(
            Path("/opt/rocm-7.2.4"),
            attest.call_args_list[0].kwargs["root"],
        )
        self.assertEqual("/usr/bin/ld.lld", environment["DL4J_ZLUDA_LINKER"])
        self.assertEqual(
            "/usr/bin/patchelf", environment["DL4J_ZLUDA_PATCHELF"]
        )
        commands = [entry.args[0] for entry in run_command.call_args_list]
        flattened = " ".join(token for command in commands for token in command)
        self.assertIn("lld", flattened)
        self.assertIn("patchelf", flattened)
        self.assertIn("rocm-hip-runtime-dev", flattened)
        self.assertIn("hsa-rocr-dev", flattened)
        self.assertIn("libnuma-dev", flattened)
        self.assertIn("libdrm-dev", flattened)
        self.assertNotIn("hsakmt-roct-dev", flattened)
        self.assertIn("rocblas-dev", flattened)
        hsakmt.assert_called_once()
        self.assertIn("hipblaslt-dev", flattened)
        self.assertIn("rocsparse-dev", flattened)
        self.assertIn("rocm-smi-lib", flattened)
        self.assertIn("miopen-hip-dev", flattened)
        self.assertNotIn("amdgpu-dkms", flattened)
        self.assertNotIn("rocminfo", flattened)

    def test_zluda_download_retries_transient_open_failure(self):
        request = build_platform.urllib.request.Request("https://example.invalid/zluda")
        response = BytesIO(b"payload")
        with patch.object(
                build_platform.urllib.request,
                "urlopen",
                side_effect=[urllib.error.URLError("temporary"), response],
        ) as urlopen, patch.object(build_platform.time, "sleep") as sleep:
            with build_platform.urlopen_with_retry(request, "ZLUDA test asset") as opened:
                self.assertEqual(b"payload", opened.read())
        self.assertEqual(2, urlopen.call_count)
        sleep.assert_called_once_with(1.0)

    def test_buildnativeoperations_rejects_unknown_zluda_target(self):
        root = Path(__file__).parents[2]
        result = subprocess.run(
            ["bash", str(root / "libnd4j/buildnativeoperations.sh"), "--zluda", "rocm6"],
            cwd=root / "libnd4j",
            capture_output=True,
            text=True,
        )
        self.assertEqual(2, result.returncode)
        self.assertIn("Unsupported --zluda target 'ROCM6'", result.stdout + result.stderr)

    def test_buildnativeoperations_accepts_explicit_zluda_off(self):
        root = Path(__file__).parents[2]
        source = (root / "libnd4j/buildnativeoperations.sh").read_text(encoding="utf-8")
        self.assertRegex(source, r'case "\$ZLUDA" in\s+OFF\)')
        self.assertIn("expected OFF, ON, or AMD", source)
        self.assertNotIn("expected OFF, ON, AMD, INTEL, or AUTO", source)

    def test_buildnativeoperations_propagates_zluda_to_canonical_cmake_configure(self):
        root = Path(__file__).parents[2]
        source = (root / "libnd4j/buildnativeoperations.sh").read_text(encoding="utf-8")
        start = source.index("run_cmake_configure() {")
        end = source.index("\n}", start)
        configure_function = source[start:end]

        self.assertIn("$ZLUDA_CMAKE", configure_function)

    def test_buildnativeoperations_keeps_parser_header_path_portable(self):
        root = Path(__file__).parents[2]
        source = (root / "libnd4j/buildnativeoperations.sh").read_text(encoding="utf-8")
        start = source.index("# CMake runs from this chip's build directory")
        end = source.index('EXPERIMENTAL_ARG=""', start)
        path_contract = source[start:end]

        self.assertIn('OP_OUTPUT_FILE_PATH="$OP_OUTPUT_FILE"', path_contract)
        self.assertIn('cygpath -m "$OP_OUTPUT_FILE_PATH"', path_contract)
        self.assertIn(
            'OP_OUTPUT_FILE_ARG="-DOP_OUTPUT_FILE=\\\"${OP_OUTPUT_FILE_PATH}\\\""',
            path_contract,
        )
        self.assertNotIn(
            'OP_OUTPUT_FILE_PATH="${BUILD_DIR}/${OP_OUTPUT_FILE}"',
            path_contract,
        )

    def test_backend_helpers_activate_backend_owned_platform_profiles(self):
        root = Path(__file__).parents[2]
        namespace = {"m": "http://maven.apache.org/POM/4.0.0"}

        def activation_property(pom_path, profile_id):
            profiles = ET.parse(pom_path).getroot().findall("m:profiles/m:profile", namespace)
            for profile in profiles:
                if profile.findtext("m:id", default="", namespaces=namespace) == profile_id:
                    return profile.findtext(
                        "m:activation/m:property/m:name",
                        default="",
                        namespaces=namespace,
                    )
            self.fail(f"profile {profile_id!r} was not found in {pom_path}")

        native_platform = (
            root
            / "nd4j/nd4j-backends/nd4j-backend-impls/nd4j-native-platform/pom.xml"
        )
        cuda_platform = (
            root
            / "nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda-platform/pom.xml"
        )
        self.assertEqual("libnd4j.helper", activation_property(native_platform, "onednn"))
        self.assertEqual("libnd4j.helper", activation_property(cuda_platform, "cudnn"))

        cpu_preset = (
            root
            / "nd4j/nd4j-backends/nd4j-backend-impls/nd4j-native-preset/"
            "src/main/java/org/nd4j/presets/cpu/Nd4jCpuPresets.java"
        ).read_text(encoding="utf-8")
        for extension in ("-onednn", "-onednn-avx2", "-onednn-avx512"):
            self.assertIn(f'"{extension}"', cpu_preset)
        self.assertNotIn('"-onednn-onednn"', cpu_preset)

    def test_avx512_classifier_enables_avx512_codegen(self):
        root = Path(__file__).parents[2]
        source = (root / "libnd4j/cmake/CompilerOptimizations.cmake").read_text(
            encoding="utf-8"
        )
        start = source.index('if(SD_EXTENSION MATCHES "avx512")')
        end = source.index("endif()", start)
        avx512_block = source[start:end]

        self.assertIn("set(CMAKE_CXX_FLAGS", avx512_block)
        for flag in (
            "-mavx512f",
            "-mavx512vl",
            "-mavx512bw",
            "-mavx512dq",
            "-mavx512cd",
        ):
            self.assertIn(flag, avx512_block)
        self.assertNotIn("SD_AVX512_FLAGS", avx512_block)
        self.assertNotIn("runtime feature detection", avx512_block)

    def test_nnapi_capability_reaches_object_compilation_and_linking(self):
        root = Path(__file__).parents[2]
        source = (root / "libnd4j/cmake/MainBuildFlow.cmake").read_text(
            encoding="utf-8"
        )
        object_start = source.index("function(create_and_link_library)")
        object_end = source.index("# OpenMP: add compile flags", object_start)
        object_setup = source[object_start:object_end]
        cpu_link_start = source.index("function(configure_cpu_linking")
        cpu_link_end = source.index("# --- Multi-Helper Library Linking", cpu_link_start)
        cpu_linking = source[cpu_link_start:cpu_link_end]

        self.assertIn("if(HAVE_NNAPI)", object_setup)
        self.assertIn(
            "target_compile_definitions(${OBJECT_LIB_NAME} PUBLIC HAVE_NNAPI=1)",
            object_setup,
        )
        self.assertIn("find_library(_sd_cpu_nnapi_library neuralnetworks)", cpu_linking)
        self.assertIn(
            'target_link_libraries(${main_target_name} PUBLIC "${_sd_cpu_nnapi_library}")',
            cpu_linking,
        )

    def test_nnapi_source_graph_excludes_cuda_zluda_and_hip_translation_units(self):
        root = Path(__file__).parents[2]
        source = (root / "libnd4j/cmake/MainBuildFlow.cmake").read_text(
            encoding="utf-8"
        )
        cpu_branch_start = source.index(
            "    else()\n        file(GLOB_RECURSE EXEC_SOURCES",
            source.index("elseif(SD_VULKAN)"),
        )
        cpu_branch_end = source.index(
            "# --- Multi-Helper Source Collection (CPU Build)", cpu_branch_start
        )
        cpu_sources = source[cpu_branch_start:cpu_branch_end]

        self.assertIn("./include/memory/impl/*.cpp", cpu_sources)
        self.assertIn("./include/memory/cpu/*.cpp", cpu_sources)
        self.assertNotIn(
            "file(GLOB_RECURSE MEMORY_SOURCES ./include/memory/*.cpp)",
            cpu_sources,
        )
        self.assertIn("if(NOT SD_CUDA)", source)
        self.assertIn("cuda|gpu|hip", source)
        self.assertIn(
            "Non-CUDA source boundary admitted CUDA/ZLUDA/HIP source", source
        )

    def test_nnapi_backend_uses_current_array_api_and_android_symbol_guard(self):
        root = Path(__file__).parents[2]
        source = (
            root / "libnd4j/include/graph/cpu/NnapiGraphBackend.cpp"
        ).read_text(encoding="utf-8")

        self.assertIn("case DataType::HALF:", source)
        self.assertNotIn("DataType::FLOAT16", source)
        self.assertNotIn("isContiguous()", source)
        self.assertIn(
            "shape::strideDescendingCAscendingF(arr->shapeInfo())",
            source,
        )
        # The backend performs the contiguity conversion once while compiling
        # operands and once again when binding runtime inputs. Outputs are
        # always staged into their own contiguous NNAPI buffers, so there are
        # two array-level guards by design.
        self.assertGreaterEqual(source.count("!isDenseCOrder(arr)"), 2)

        # Older NNAPI paths used the API-28 relax-computation entry point.
        # Current code no longer emits that optional call; if a future backend
        # restores it, require the Android API guard around the call.
        call = source.find("ANeuralNetworksModel_relaxComputationFloat32toFloat16")
        if call >= 0:
            compile_guard = source.rfind(
                "#if defined(__ANDROID_API__) && __ANDROID_API__ >= 28",
                0,
                call,
            )
            self.assertGreaterEqual(compile_guard, 0)
            self.assertGreater(source.index("#endif", call), call)

    def test_shared_native_script_emits_specialized_classifiers(self):
        root = Path(__file__).parents[2]
        script = root / "build-scripts/release/native-platform.sh"

        def command(**environment):
            process_environment = {"PATH": "/usr/bin:/bin", "DL4J_BUILD_THREADS": "16", "DL4J_MVN_FLAGS": ""}
            process_environment.update(environment)
            result = subprocess.run(["bash", str(script), "--print"], env=process_environment,
                                    check=True, capture_output=True, text=True)
            return shlex.split(result.stdout)

        cuda = command(DL4J_FAMILY="linux-cuda", DL4J_CUDA_VERSION="12.9", DL4J_HELPER="compile")
        self.assertIn("-Dlibnd4j.classifier=linux-x86_64-cuda-12.9-compile", cuda)
        self.assertIn("-Djavacpp.platform.extension=-compile", cuda)

        windows_onednn = command(DL4J_FAMILY="windows-cpu", DL4J_HELPER="onednn")
        self.assertIn("-Dlibnd4j.helper=onednn", windows_onednn)
        self.assertIn("-Djavacpp.platform.extension=-onednn", windows_onednn)
        self.assertIn("-Dlibnd4j.classifier=windows-x86_64-onednn", windows_onednn)
        self.assertNotIn("-Dlibnd4j.extension=onednn", windows_onednn)
        self.assertFalse(any("onednn-onednn" in argument for argument in windows_onednn))

        windows_cpu = command(DL4J_FAMILY="windows-cpu")
        self.assertIn("-Djavacpp.platform.build=windows-x86_64-mingw", windows_cpu)
        self.assertIn("-Djavacpp.platform.properties=windows-x86_64-mingw", windows_cpu)
        self.assertIn("-Djavacpp.platform.compiler=g++", windows_cpu)

        windows_cudnn = command(
            DL4J_FAMILY="windows-cuda",
            DL4J_CUDA_VERSION="12.9",
            DL4J_HELPER="cudnn",
        )
        self.assertIn("-Dlibnd4j.helper=cudnn", windows_cudnn)
        self.assertIn("-Djavacpp.platform.extension=-cudnn", windows_cudnn)
        self.assertIn(
            "-Dlibnd4j.classifier=windows-x86_64-cuda-12.9-cudnn",
            windows_cudnn,
        )
        self.assertNotIn("-Dlibnd4j.extension=cudnn", windows_cudnn)

        windows_vulkan = command(DL4J_FAMILY="windows-vulkan")
        self.assertIn("-Pvulkan", windows_vulkan)
        self.assertIn("-Dlibnd4j.vulkan", windows_vulkan)
        self.assertIn("-Djavacpp.platform=windows-x86_64", windows_vulkan)
        self.assertIn("-Djavacpp.platform.build=windows-x86_64-mingw", windows_vulkan)
        self.assertIn(
            "-Djavacpp.platform.properties=windows-x86_64-mingw", windows_vulkan
        )
        self.assertIn("-Djavacpp.platform.compiler=g++", windows_vulkan)
        self.assertIn("-Dlibnd4j.platform=windows-x86_64", windows_vulkan)
        self.assertIn("-Dplatform.classifier=windows-x86_64", windows_vulkan)
        self.assertIn("-Dlibnd4j.classifier=windows-x86_64", windows_vulkan)
        self.assertIn("-Dlibnd4j.triton=ON", windows_vulkan)
        self.assertIn("-Dlibnd4j.mlir=ON", windows_vulkan)
        self.assertNotIn("-Dlibnd4j.triton=OFF", windows_vulkan)
        self.assertNotIn("-Dlibnd4j.mlir=OFF", windows_vulkan)
        self.assertEqual(
            ":nd4j-vulkan,:nd4j-vulkan-preset,:nd4j-vulkan-platform,:libnd4j",
            windows_vulkan[windows_vulkan.index("-pl") + 1],
        )

        linux_vulkan = command(DL4J_FAMILY="vulkan")
        self.assertIn("-Pvulkan", linux_vulkan)
        self.assertIn("-Dlibnd4j.vulkan", linux_vulkan)
        self.assertIn("-Dlibnd4j.triton=ON", linux_vulkan)
        self.assertIn("-Dlibnd4j.mlir=ON", linux_vulkan)
        self.assertIn("-Djavacpp.platform=linux-x86_64", linux_vulkan)
        self.assertIn("-Dplatform.classifier=linux-x86_64", linux_vulkan)

        linux_vulkan_mlir = command(DL4J_FAMILY="vulkan-mlir")
        self.assertIn("-Dlibnd4j.triton=ON", linux_vulkan_mlir)
        self.assertIn("-Dlibnd4j.mlir=ON", linux_vulkan_mlir)
        self.assertIn("-Dplatform.classifier=linux-x86_64-compile", linux_vulkan_mlir)

        metal = command(DL4J_FAMILY="macos-arm64", DL4J_HELPER="mps")
        self.assertIn("-Pmetal", metal)

        android = command(
            DL4J_FAMILY="android-arm64",
            DL4J_ANDROID_API="27",
            DL4J_CMAKE_ARGS="-DSD_BUILD_WITH_JAVA=OFF",
            ANDROID_NDK="/opt/android/android-ndk-r27d",
        )
        self.assertIn("-Dlibnd4j.android.api=27", android)
        self.assertIn("-Dlibnd4j.build.with.java=OFF", android)
        self.assertIn("-Dlibnd4j.cmake=-DSD_BUILD_WITH_JAVA=OFF", android)
        self.assertIn(
            "-Djavacpp.platform.compiler=/opt/android/android-ndk-r27d/"
            "toolchains/llvm/prebuilt/linux-x86_64/bin/"
            "aarch64-linux-android27-clang++",
            android,
        )

        android_x86 = command(
            DL4J_FAMILY="android-x86_64",
            DL4J_ANDROID_API="21",
            DL4J_CMAKE_ARGS="-DSD_BUILD_WITH_JAVA=OFF",
            ANDROID_NDK="/opt/android/android-ndk-r26d",
        )
        self.assertIn("-Dlibnd4j.android.api=21", android_x86)
        self.assertIn("-Dlibnd4j.build.with.java=OFF", android_x86)

        android_vulkan = command(
            DL4J_FAMILY="android-arm64-vulkan",
            DL4J_ANDROID_API="24",
            DL4J_CMAKE_ARGS="-DSD_BUILD_WITH_JAVA=OFF",
            ANDROID_NDK="/opt/android/android-ndk-r27d",
        )
        self.assertEqual(
            ":nd4j-vulkan,:nd4j-vulkan-preset,:nd4j-vulkan-platform,:libnd4j",
            android_vulkan[android_vulkan.index("-pl") + 1],
        )

        linux_arm64 = command(DL4J_FAMILY="linux-arm64")
        self.assertNotIn("-Dlibnd4j.build.with.java=OFF", linux_arm64)
        self.assertIn("-Djavacpp.platform.compiler=g++", linux_arm64)
        arm64_libgcc = "-Dplatform.linker.flag.no.undefined=-Wl,--no-undefined,-lgcc"
        for generated_command in (linux_arm64, android, android_x86, metal, cuda):
            self.assertNotIn(arm64_libgcc, generated_command)

        cpu_preset = (
            root
            / "nd4j/nd4j-backends/nd4j-backend-impls/nd4j-native-preset/"
            "src/main/java/org/nd4j/presets/cpu/Nd4jCpuPresets.java"
        ).read_text(encoding="utf-8")
        self.assertIn(
            '@Platform(value = "linux-arm64", link = {"nd4jcpu", "dl", "gcc"},',
            cpu_preset,
        )

        for family in ("tpu", "hexagon", "vulkan"):
            accelerator = command(DL4J_FAMILY=family)
            self.assertIn(f"-P{family}", accelerator)

        vulkan = command(DL4J_FAMILY="vulkan-mlir")
        self.assertIn("-Pvulkan", vulkan)
        self.assertIn("-Dplatform.classifier=linux-x86_64-compile", vulkan)

        zluda = command(
            DL4J_FAMILY="zluda",
            DL4J_CUDA_VERSION="12.9",
            DL4J_PLATFORM_EXTENSION="-zluda",
            DL4J_CLASSIFIER="linux-x86_64-cuda-12.9-zluda",
        )
        self.assertIn("-Dlibnd4j.zluda=AMD", zluda)
        self.assertIn("-Dlibnd4j.zluda.version=v7-preview.8", zluda)
        self.assertFalse(any("zluda.root" in argument for argument in zluda))
        self.assertIn("-Dlibnd4j.classifier=linux-x86_64-cuda-12.9-zluda", zluda)
        self.assertIn("-Djavacpp.platform.extension=-zluda", zluda)
        self.assertIn("-Pzluda", zluda)
        self.assertIn("-Pzluda-platform", zluda)
        self.assertEqual(
            ":nd4j-cuda-backend-common,:nd4j-cuda-12.9-preset,:nd4j-zluda-12.9,"
            ":nd4j-zluda-12.9-platform,:libnd4j",
            zluda[zluda.index("-pl") + 1],
        )

        windows_zluda = command(
            DL4J_FAMILY="windows-zluda",
            DL4J_CUDA_VERSION="12.9",
            DL4J_PLATFORM_EXTENSION="-zluda",
            DL4J_CLASSIFIER="windows-x86_64-cuda-12.9-zluda",
            DL4J_ZLUDA_TARGET="AMD",
        )
        self.assertIn("-Dlibnd4j.classifier=windows-x86_64-cuda-12.9-zluda", windows_zluda)
        self.assertFalse(any("zluda.root" in argument for argument in windows_zluda))
        self.assertIn("-Djavacpp.platform=windows-x86_64", windows_zluda)
        self.assertIn("-Dlibnd4j.platform=windows-x86_64", windows_zluda)
        self.assertIn("-Dlibnd4j.oom.killer=OFF", windows_zluda)
        self.assertIn("-Pzluda-platform", windows_zluda)
        self.assertEqual(
            ":nd4j-cuda-backend-common,:nd4j-cuda-12.9-preset,:nd4j-zluda-12.9,"
            ":nd4j-zluda-12.9-platform,:libnd4j",
            windows_zluda[windows_zluda.index("-pl") + 1],
        )

        for rocm_version in ("7.2.4", "6.2.4"):
            extension = f"-zluda-rocm-{rocm_version}"
            for family, platform_name in (
                ("zluda", "linux-x86_64"),
                ("windows-zluda", "windows-x86_64"),
            ):
                with self.subTest(rocm=rocm_version, family=family):
                    classifier = (
                        f"{platform_name}-cuda-12.9-zluda-rocm-{rocm_version}"
                    )
                    versioned = command(
                        DL4J_FAMILY=family,
                        DL4J_CUDA_VERSION="12.9",
                        DL4J_PLATFORM_EXTENSION=extension,
                        DL4J_CLASSIFIER=classifier,
                        DL4J_ROCM_VERSION=rocm_version,
                    )
                    self.assertIn(
                        f"-Djavacpp.platform.extension={extension}", versioned
                    )
                    self.assertIn(f"-Dlibnd4j.classifier={classifier}", versioned)
                    self.assertIn(f"-Drocm.version={rocm_version}", versioned)
                    self.assertNotIn("-Djavacpp.platform.extension=-zluda", versioned)

    def test_vulkan_native_family_tracks_worker_os(self):
        build = {
            "backend": "vulkan",
            "javacppPlatform": "windows-x86_64",
            "modules": [":nd4j-vulkan"],
        }
        variant = {"name": "base"}
        self.assertEqual(
            "windows-vulkan",
            build_platform.shared_native_family(
                {"os": "windows", "build": build}, variant
            ),
        )
        build["javacppPlatform"] = "linux-x86_64"
        # The base Vulkan runtime includes MLIR/Triton, but must retain the
        # unextended linux-x86_64 classifier.  Only the explicit -compile
        # variant belongs to the vulkan-mlir family.
        variant["mlir"] = True
        self.assertEqual(
            "vulkan",
            build_platform.shared_native_family(
                {"os": "linux", "build": build}, variant
            ),
        )
        self.assertEqual(
            "vulkan-mlir",
            build_platform.shared_native_family(
                {"os": "linux", "build": build},
                {"name": "compile", "suffix": "-compile", "mlir": True},
            ),
        )
        self.assertEqual(
            ("nd4j-vulkan",),
            build_platform.required_classifier_artifact_ids(
                build,
                {
                    "mode": "classifier",
                    "artifactIds": [
                        "nd4j-vulkan",
                        "nd4j-vulkan-preset",
                        "nd4j-vulkan-platform",
                    ],
                },
            ),
        )

    def test_zluda_native_family_tracks_worker_os(self):
        build = {"zludaVersion": "v6"}
        variant = {"name": "zluda"}
        self.assertEqual(
            "zluda",
            build_platform.shared_native_family({"os": "linux", "build": build}, variant),
        )
        self.assertEqual(
            "windows-zluda",
            build_platform.shared_native_family({"os": "windows", "build": build}, variant),
        )

    def test_zluda_native_driver_passes_exact_release_classifier_contract(self):
        variant = {
            "name": "zluda",
            "classifierSuffix": "-cuda-12.9-zluda-rocm-6.2.4",
            "platformExtension": "-zluda-rocm-6.2.4",
        }
        shard = {
            "id": "linux-x86_64-zluda-rocm-6.2.4",
            "os": "linux",
            "workloads": ["maven"],
            "build": {
                "backend": "cuda",
                "cudaVersion": "12.9",
                "javacppPlatform": "linux-x86_64",
                "zludaVersion": "v6",
                "mavenArgs": ["-Dlibnd4j.zluda=AMD"],
                "modules": [],
                "variants": [variant],
            },
        }
        calls = []
        with patch.object(build_platform, "prepare_openblas"), patch.object(
            build_platform,
            "run",
            side_effect=lambda command, _cwd, env: calls.append((command, env.copy())),
        ):
            build_platform.build_native_platform(
                Path("/source"), shard, Path("/m2"), {}, None
            )
        invocation = calls[0][1]
        self.assertEqual("-zluda-rocm-6.2.4", invocation["DL4J_PLATFORM_EXTENSION"])
        self.assertEqual("v6", invocation["DL4J_ZLUDA_VERSION"])
        self.assertEqual(
            "linux-x86_64-cuda-12.9-zluda-rocm-6.2.4",
            invocation["DL4J_CLASSIFIER"],
        )

    def test_aws_cuda_compile_variant_uses_workflow_compile_classifier_path(self):
        shard = {
            "id": "linux-x86_64-cuda-12-9", "os": "linux", "workloads": ["maven"],
            "build": {"backend": "cuda", "cudaVersion": "12.9", "javacppPlatform": "linux-x86_64",
                      "modules": [], "variants": [{"name": "compile", "triton": True}]},
        }
        calls = []
        with patch.object(build_platform, "prepare_openblas"), \
                patch.object(build_platform, "run", side_effect=lambda command, _cwd, env: calls.append((command, env))):
            build_platform.build_native_platform(Path("/source"), shard, Path("/m2"), {}, None)
        self.assertEqual("compile", calls[0][1]["DL4J_HELPER"])
        self.assertEqual("12.9", calls[0][1]["DL4J_CUDA_VERSION"])

    def test_aws_cross_platform_invokes_the_shared_workflow_script(self):
        calls = []
        build = {"javacppPlatform": "linux-x86_64"}
        with patch.object(build_platform, "run", side_effect=lambda command, _cwd, env: calls.append((command, env))):
            build_platform.build_cross_platform(Path("/source"), build, Path("/m2"), {})
        self.assertEqual(["--run-tokenizers", "--run-java"], [call[0][-1] for call in calls])
        self.assertTrue(all(Path(call[0][1]).name == "cross-platform.sh" for call in calls))
        self.assertTrue(all(call[1]["DL4J_MAVEN_GOAL"] == "install" for call in calls))

    def test_sdx_release_component_is_opt_in_and_publishable(self):
        build = {
            "backend": "cpu",
            "javacppPlatform": "windows-x86_64",
            "profiles": ["cpu", "sdx"],
            "modules": [":nd4j-native", ":nd4j-native-preset"],
        }
        rules = {
            "mode": "classifier",
            "artifactIds": ["nd4j-native", "nd4j-native-preset"],
            "classifierTokens": ["windows-x86_64"],
        }
        self.assertTrue(build_platform.sdx_enabled_for_build(build))
        build_platform.enable_sdx_release_component(build, rules)
        self.assertEqual(
            {
                ":nd4j-sdx-preset",
                ":nd4j-sdx-model",
                ":nd4j-sdx",
                ":nd4j-sdx-litertlm",
            },
            set(build["modules"]) & set(build_platform.SDX_MODULES),
        )
        self.assertEqual(
            set(build_platform.SDX_ARTIFACT_IDS),
            set(rules["unclassifiedArtifactIds"]),
        )
        self.assertIn("nd4j-sdx", build_platform.required_classifier_artifact_ids(build, rules))

        non_sdx = dict(build, profiles=["cpu"])
        self.assertFalse(build_platform.sdx_enabled_for_build(non_sdx))

    def test_sdx_gpu_backends_use_distinct_classifier_names(self):
        cuda = {
            "backend": "cuda",
            "cudaVersion": "12.9",
            "javacppPlatform": "linux-x86_64",
            "profiles": ["cuda", "sdx"],
        }
        vulkan = {
            "backend": "vulkan",
            "javacppPlatform": "linux-x86_64",
            "profiles": ["vulkan", "sdx"],
        }
        zluda = dict(
            cuda,
            zludaVersion="v6",
            rocmVersion="6.2.4",
        )
        variant = {"name": "base", "platformExtension": ""}
        self.assertTrue(build_platform.sdx_enabled_for_build(cuda))
        self.assertTrue(build_platform.sdx_enabled_for_build(vulkan))
        self.assertTrue(build_platform.sdx_enabled_for_build(zluda))
        self.assertEqual(
            "linux-x86_64-cuda-12.9",
            build_platform.sdx_variant_artifact_classifier(cuda, variant),
        )
        self.assertEqual(
            "linux-x86_64-vulkan",
            build_platform.sdx_variant_artifact_classifier(vulkan, variant),
        )
        self.assertEqual(
            "linux-x86_64",
            build_platform.sdx_variant_artifact_classifier(zluda, variant),
        )

    def test_native_sdx_command_contains_profile_and_modules(self):
        root = Path(__file__).parents[2]
        script = root / "build-scripts/release/native-platform.sh"
        env = os.environ.copy()
        env.update({
            "DL4J_FAMILY": "windows-cpu",
            "DL4J_BUILD_THREADS": "4",
            "DL4J_BUILD_SDX": "1",
            "DL4J_MAVEN_GOAL": "install",
        })
        command = subprocess.check_output(
            [str(script), "--print"], cwd=root, env=env, text=True
        )
        self.assertIn("-Psdx", command)
        for module in ("nd4j-sdx-model", "nd4j-sdx"):
            self.assertIn(module, command)

    def test_native_sdx_gpu_command_contains_backend_configuration(self):
        root = Path(__file__).parents[2]
        script = root / "build-scripts/release/native-platform.sh"
        env = os.environ.copy()
        env.update({
            "DL4J_FAMILY": "windows-vulkan",
            "DL4J_BUILD_THREADS": "4",
            "DL4J_BUILD_SDX": "1",
            "DL4J_SDX_NATIVE_LIBRARY": "nd4jvulkan",
            "DL4J_SDX_PLATFORM_LINKS": "nd4jvulkan",
            "DL4J_SDX_OUTPUT_PATH": "/tmp/vulkan",
            "DL4J_SDX_CLASSIFIER": "windows-x86_64-vulkan",
            "DL4J_MAVEN_GOAL": "install",
        })
        command = subprocess.check_output(
            [str(script), "--print"], cwd=root, env=env, text=True
        )
        self.assertIn("-Psdx", command)
        self.assertIn("nd4j-sdx", command)
        self.assertIn("-Dsdx.native.library=nd4jvulkan", command)
        self.assertIn("-Dsdx.platform.classifier=windows-x86_64-vulkan", command)

    def test_sdx_gnu_linker_flag_is_not_active_on_macos(self):
        root = Path(__file__).parents[2]
        namespace = {"m": "http://maven.apache.org/POM/4.0.0"}
        pom = ET.parse(
            root
            / "nd4j/nd4j-backends/nd4j-backend-impls/nd4j-sdx/pom.xml"
        ).getroot()
        self.assertEqual(
            "",
            pom.findtext(
                "m:properties/m:javacpp.compiler.options", namespaces=namespace
            )
            or "",
        )
        profiles = {
            profile.findtext("m:id", namespaces=namespace): profile
            for profile in pom.findall("m:profiles/m:profile", namespace)
        }
        unix_flags = profiles["os-unix-linker-flags"]
        self.assertEqual(
            "unix",
            unix_flags.findtext("m:activation/m:os/m:family", namespaces=namespace),
        )
        self.assertEqual(
            "!mac os x",
            unix_flags.findtext("m:activation/m:os/m:name", namespaces=namespace),
        )
        self.assertEqual(
            "-Wl,--no-undefined",
            unix_flags.findtext(
                "m:properties/m:javacpp.compiler.options", namespaces=namespace
            ),
        )

    def test_triton_cpu_patch_uses_raw_bits_when_gcc_lacks_float16(self):
        root = Path(__file__).parents[2]
        patch_script = (root / "libnd4j/cmake/patch_triton_cpu.cmake").read_text(
            encoding="utf-8"
        )
        dependencies = (root / "libnd4j/cmake/Dependencies.cmake").read_text(
            encoding="utf-8"
        )
        llvm_patch = (
            root / "libnd4j/cmake/patch_external_llvm_coexistence.cmake"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'string(REPLACE "#ifdef FLT16_MAX" "#ifdef DL4J_NATIVE_FLOAT16"',
            patch_script,
        )
        self.assertIn("#if defined(__clang__)", patch_script)
        self.assertIn("defined(__GNUC__) && defined(__aarch64__)", patch_script)
        self.assertIn("typedef uint16_t fp16_t", patch_script)
        self.assertIn("managed-llvm-patches-v12", dependencies)
        self.assertIn("_sd_mlir_direct_options_anchor", llvm_patch)
        self.assertIn("_sd_mlir_legacy_options_anchor", llvm_patch)
        self.assertIn("SD_ANDROID_MLIR_EXECUTION_ENGINE_V2", llvm_patch)
        self.assertIn("unsupported MLIR execution-engine option layout", llvm_patch)

    def test_external_llvm_patch_forces_android_execution_engine_after_arch_check(self):
        root = Path(__file__).parents[2]
        patch_script = root / "libnd4j/cmake/patch_external_llvm_coexistence.cmake"
        direct_layout = """if(${LLVM_NATIVE_ARCH} IN_LIST LLVM_TARGETS_TO_BUILD)
  set(MLIR_ENABLE_EXECUTION_ENGINE 1)
else()
  set(MLIR_ENABLE_EXECUTION_ENGINE 0)
endif()"""
        legacy_layout = """if(${LLVM_NATIVE_ARCH} IN_LIST LLVM_TARGETS_TO_BUILD)
  set(MLIR_ENABLE_EXECUTION_ENGINE_default 1)
else()
  set(MLIR_ENABLE_EXECUTION_ENGINE_default 0)
endif()
option(MLIR_ENABLE_EXECUTION_ENGINE
       \"Enable building the MLIR Execution Engine.\"
       ${MLIR_ENABLE_EXECUTION_ENGINE_default})"""
        llvm_dylib_option = """cmake_dependent_option(LLVM_BUILD_LLVM_DYLIB \"Build libllvm dynamic library\" ${LLVM_BUILD_LLVM_DYLIB_default}
                       \"CAN_BUILD_LLVM_DYLIB\" OFF)
"""

        for layout in (direct_layout, legacy_layout):
            with self.subTest(layout=layout.splitlines()[1]):
                with tempfile.TemporaryDirectory() as temp_dir:
                    source_dir = Path(temp_dir)
                    handle_options = source_dir / "llvm/cmake/modules/HandleLLVMOptions.cmake"
                    handle_options.parent.mkdir(parents=True)
                    handle_options.write_text("include(LLVMProcessSources)\n", encoding="utf-8")
                    llvm_cmake = source_dir / "llvm/CMakeLists.txt"
                    llvm_cmake.write_text(llvm_dylib_option, encoding="utf-8")
                    mlir_cmake = source_dir / "mlir/CMakeLists.txt"
                    mlir_cmake.parent.mkdir(parents=True)
                    mlir_cmake.write_text(layout + "\n", encoding="utf-8")

                    command = [
                        "cmake",
                        f"-DSOURCE_DIR={source_dir}",
                        "-DSD_EXTERNAL_PROJECT=LLVM",
                        "-P",
                        str(patch_script),
                    ]
                    subprocess.run(command, check=True, capture_output=True, text=True)
                    subprocess.run(command, check=True, capture_output=True, text=True)

                    patched = mlir_cmake.read_text(encoding="utf-8")
                    marker = "# SD_ANDROID_MLIR_EXECUTION_ENGINE_V2"
                    self.assertEqual(1, patched.count(marker))
                    self.assertGreater(patched.index(marker), patched.index(layout))

    def test_windows_cross_platform_tokenizer_command_selects_mingw_properties(self):
        root = Path(__file__).parents[2]
        environment = {
            "DL4J_PLATFORM": "windows-x86_64",
            "DL4J_OS": "windows",
            "PATH": "/usr/bin:/bin",
        }
        result = subprocess.run(
            ["bash", str(root / "build-scripts/release/cross-platform.sh"), "--print-tokenizers"],
            env=environment, check=True, capture_output=True, text=True,
        )
        command = shlex.split(result.stdout)
        self.assertIn("-Djavacpp.platform.build=windows-x86_64-mingw", command)
        self.assertIn("-Djavacpp.platform.properties=windows-x86_64-mingw", command)
        self.assertIn("-Djavacpp.platform.compiler=g++", command)

        java_result = subprocess.run(
            ["bash", str(root / "build-scripts/release/cross-platform.sh"), "--print-java"],
            env=environment, check=True, capture_output=True, text=True,
        )
        java_command = shlex.split(java_result.stdout)
        self.assertIn("-Djavacpp.platform.build=windows-x86_64-mingw", java_command)
        self.assertIn("-Djavacpp.platform.properties=windows-x86_64-mingw", java_command)
        self.assertIn("-Djavacpp.platform.compiler=g++", java_command)

    def test_linux_cross_platform_java_command_does_not_infer_accelerator_profiles(self):
        root = Path(__file__).parents[2]
        environment = {
            "DL4J_PLATFORM": "linux-x86_64",
            "DL4J_OS": "linux",
            "PATH": "/usr/bin:/bin",
        }
        result = subprocess.run(
            ["bash", str(root / "build-scripts/release/cross-platform.sh"), "--print-java"],
            env=environment, check=True, capture_output=True, text=True,
        )
        command = shlex.split(result.stdout)
        self.assertEqual([], [argument for argument in command if argument.startswith("-P")])

    def test_update_versions_propagates_maven_failure(self):
        root = Path(__file__).parents[2]
        script = root / "update-versions.sh"
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            pom = project / "pom.xml"
            pom.write_text("<project><version>1.0.0-SNAPSHOT</version></project>\n", encoding="utf-8")
            bin_directory = project / "bin"
            bin_directory.mkdir()
            fake_maven = bin_directory / "mvn"
            fake_maven.write_text("#!/usr/bin/env bash\nexit 23\n", encoding="utf-8")
            fake_maven.chmod(0o755)
            environment = os.environ.copy()
            environment["PATH"] = f"{bin_directory}:{environment['PATH']}"
            result = subprocess.run(
                ["bash", str(script), "1.0.0-SNAPSHOT", "1.0.0"],
                cwd=project, env=environment, capture_output=True, text=True,
            )
            self.assertEqual(23, result.returncode)
            self.assertIn("<version>1.0.0-SNAPSHOT</version>", pom.read_text(encoding="utf-8"))

    def test_shared_linux_script_emits_the_workflow_command(self):
        root = Path(__file__).parents[2]
        environment = {
            "DL4J_HELPER": "", "DL4J_EXTENSION": "", "DL4J_LIBND4J_FILE_DOWNLOAD": "",
            "DL4J_BUILD_THREADS": "16", "DL4J_MATRIX_MVN_EXT": "", "PATH": "/usr/bin:/bin",
        }
        result = subprocess.run(
            ["bash", str(root / "build-scripts/release/linux-x86_64.sh"), "--print"],
            env=environment, check=True, capture_output=True, text=True,
        )
        command = shlex.split(result.stdout)
        self.assertEqual("mvn", command[0])
        self.assertIn("-X", command)
        self.assertEqual(":nd4j-native,:nd4j-native-preset,:libnd4j", command[command.index("-pl") + 1])
        self.assertEqual("deploy", command[-2])
        self.assertEqual("-DskipTests", command[-1])

    def test_github_and_clouds_reference_the_same_release_worker(self):
        root = Path(__file__).parents[2]
        linux_workflow = (root / ".github/workflows/build-deploy-linux-x86_64.yml").read_text(encoding="utf-8")
        cross_workflow = (root / ".github/workflows/build-deploy-cross-platform.yml").read_text(encoding="utf-8")
        reusable = (root / ".github/workflows/_release-worker.yml").read_text(encoding="utf-8")
        action = (root / ".github/actions/run-release-worker/action.yml").read_text(encoding="utf-8")
        driver = (root / "release/aws/build-platform.py").read_text(encoding="utf-8")
        self.assertIn(".github/workflows/_release-worker.yml", linux_workflow)
        self.assertIn(".github/workflows/_release-worker.yml", cross_workflow)
        self.assertIn("release/github/prepare-worker.py matrix", reusable)
        self.assertIn("release/aws/build-platform.py", action)
        self.assertIn('script_name = "linux-x86_64.sh"', driver)
        self.assertIn('source / "build-scripts/release/cross-platform.sh"', driver)
        self.assertIn('else "native-platform.sh"', driver)

    def test_omnihub_pipeline_dependencies_are_in_the_shared_java_reactor(self):
        root = Path(__file__).parents[2]
        namespace = {"m": "http://maven.apache.org/POM/4.0.0"}
        nd4j_pom = ET.parse(root / "nd4j/pom.xml").getroot()
        reactor_modules = {
            module.text for module in nd4j_pom.findall("./m:modules/m:module", namespace)
        }
        omnihub_pom = ET.parse(root / "omnihub/pom.xml").getroot()
        pipeline_dependencies = {
            dependency.find("m:artifactId", namespace).text
            for dependency in omnihub_pom.findall("./m:dependencies/m:dependency", namespace)
            if dependency.find("m:artifactId", namespace) is not None
            and dependency.find("m:artifactId", namespace).text.startswith("samediff-pipeline-")
        }

        self.assertIn("samediff-pipeline-ggml", pipeline_dependencies)
        self.assertTrue(
            pipeline_dependencies.issubset(reactor_modules),
            f"OmniHub pipeline dependencies missing from nd4j reactor: {pipeline_dependencies - reactor_modules}",
        )
        for module in pipeline_dependencies:
            self.assertTrue((root / "nd4j" / module / "pom.xml").is_file(), module)

        service = root / "nd4j/samediff-pipeline-ggml/src/main/resources/META-INF/services" \
            / "org.eclipse.deeplearning4j.pipeline.PipelineLoader"
        self.assertEqual(
            "org.eclipse.deeplearning4j.ggml.GGMLPipelineLoader",
            service.read_text(encoding="utf-8").strip(),
        )

    def test_bootstrap_and_workers_emit_durable_lifecycle_phases(self):
        bootstrap = release.bootstrap_user_data("linux", "https://example.invalid/worker")
        self.assertIn("phase=cloud-init status=started", bootstrap)
        self.assertIn("phase=worker-download status=complete", bootstrap)
        self.assertIn("export HOME=${HOME:-/root}", bootstrap)
        self.assertLess(bootstrap.index("export HOME="), bootstrap.index("curl --fail"))
        root = Path(__file__).parent
        linux = (root / "worker.sh").read_text(encoding="utf-8")
        windows = (root / "worker.ps1").read_text(encoding="utf-8")
        for phase in ("logging-prerequisites", "toolchain-packages", "source-checkout", "matrix-build", "artifact-packaging", "finalize"):
            self.assertIn(phase, linux)
            self.assertIn(phase, windows)
        self.assertLess(linux.index("start_log_forwarder\n"), linux.index("phase toolchain-packages started"))
        self.assertLess(windows.index("Start-LogForwarder\n"), windows.index("Write-Phase 'toolchain-packages' 'started'"))
        self.assertLess(linux.index("export HOME="), linux.index("CONFIG_B64="))
        self.assertLess(linux.index("export CARGO_HOME="), linux.index("rust-toolchain started"))
        self.assertIn('SCCACHE_ROOT=${WORK_ROOT}/sccache', linux)
        self.assertIn('DL4J_SCCACHE_DIR=/dl4j-sccache-root/cache', linux)
        self.assertIn('${SCCACHE_ROOT}:/dl4j-sccache-root', linux)
        self.assertIn('DL4J_SCCACHE_DIR=/github/sccache-root/cache', linux)
        self.assertIn('${SCCACHE_ROOT}:/github/sccache-root', linux)
        self.assertNotIn('${HOME}/.cargo', linux)
        self.assertLess(windows.index("$env:CARGO_HOME ="), windows.index("rustup toolchain install"))
        self.assertNotIn("$env:USERPROFILE", windows)

    def test_unix_worker_initializes_cloud_init_environment_under_strict_mode(self):
        worker = (Path(__file__).parent / "worker.sh").read_text(encoding="utf-8")
        environment_prefix = worker.split("CONFIG_B64=", 1)[0]
        probe = environment_prefix + "printf '%s\\n' \"$HOME|$USER|$LOGNAME|$PATH\""
        result = subprocess.run(
            ["bash", "-u", "-c", probe], env={}, check=True, capture_output=True, text=True
        )
        home, user, logname, path = result.stdout.strip().split("|", 3)
        self.assertEqual("/root", home)
        self.assertTrue(user)
        self.assertEqual(user, logname)
        self.assertIn("/usr/bin", path)


class EnvironmentWizardTest(unittest.TestCase):
    class Credentials:
        def __init__(self, method):
            self.method = method

    class Sts:
        def __init__(self, usable):
            self.usable = usable

        def get_caller_identity(self):
            if not self.usable:
                raise AssertionError("STS must not run without resolved credentials")
            return {"Account": "123456789012", "Arn": "arn:aws:iam::123456789012:user/test"}

    class Session:
        available_profiles = ["release"]

        def __init__(self, **kwargs):
            self.region_name = kwargs.get("region_name") or os.environ.get("AWS_REGION")
            access_key = kwargs.get("aws_access_key_id") or os.environ.get("AWS_ACCESS_KEY_ID")
            secret_key = kwargs.get("aws_secret_access_key") or os.environ.get("AWS_SECRET_ACCESS_KEY")
            profile = kwargs.get("profile_name") or os.environ.get("AWS_PROFILE")
            self.method = "env" if access_key and secret_key else ("shared-credentials-file" if profile == "release" else None)

        def get_credentials(self):
            return EnvironmentWizardTest.Credentials(self.method) if self.method else None

        def client(self, name):
            if name == "sts":
                return EnvironmentWizardTest.Sts(bool(self.method))
            return object()

    class Boto3:
        @staticmethod
        def Session(**kwargs):
            return EnvironmentWizardTest.Session(**kwargs)

    def test_ci_never_enables_wizard_even_with_a_tty(self):
        tty = type("Tty", (), {"isatty": lambda self: True})()
        with patch.dict(os.environ, {"CI": "true"}, clear=True), patch.object(release.sys, "stdin", tty):
            self.assertFalse(release.interactive_wizard_enabled(True))

    def test_region_validation_is_partition_agnostic(self):
        self.assertTrue(release.valid_aws_region("us-east-1"))
        self.assertTrue(release.valid_aws_region("eusc-de-east-1"))
        self.assertFalse(release.valid_aws_region("global"))

    def test_noninteractive_missing_configuration_fails_with_recovery_command(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(release, "_boto3", return_value=(self.Boto3, Exception)),
            self.assertRaisesRegex(SystemExit, "release/aws/release.py configure"),
        ):
            release.session_clients(allow_wizard=False)

    def test_existing_standard_environment_does_not_prompt(self):
        environment = {
            "AWS_REGION": "us-east-1",
            "AWS_ACCESS_KEY_ID": "AKIATEST",
            "AWS_SECRET_ACCESS_KEY": "already-configured",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(release, "_boto3", return_value=(self.Boto3, Exception)),
            patch("builtins.input", side_effect=AssertionError("wizard unexpectedly prompted")),
        ):
            session, region, *_ = release.session_clients(allow_wizard=True)
        self.assertEqual("us-east-1", region)
        self.assertEqual("env", session.get_credentials().method)

    def test_existing_standard_profile_does_not_prompt(self):
        environment = {"AWS_REGION": "us-east-1", "AWS_PROFILE": "release"}
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(release, "_boto3", return_value=(self.Boto3, Exception)),
            patch("builtins.input", side_effect=AssertionError("wizard unexpectedly prompted")),
        ):
            session, _, *_ = release.session_clients(allow_wizard=True)
        self.assertEqual("shared-credentials-file", session.get_credentials().method)

    def test_wizard_sets_region_and_hidden_access_credentials_for_current_command(self):
        stderr = StringIO()
        stdout = StringIO()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(release, "_boto3", return_value=(self.Boto3, Exception)),
            patch.object(release, "interactive_wizard_enabled", return_value=True),
            patch("builtins.input", side_effect=["us-east-1", "AKIAWIZARD"]),
            patch.object(release.getpass, "getpass", side_effect=["never-print-this", ""]) as secret_input,
            redirect_stderr(stderr),
            redirect_stdout(stdout),
        ):
            session, region, *_ = release.session_clients()
            self.assertNotIn("AWS_ACCESS_KEY_ID", os.environ)
            self.assertNotIn("AWS_SECRET_ACCESS_KEY", os.environ)
            self.assertNotIn("AWS_PROFILE", os.environ)
            self.assertEqual("us-east-1", os.environ["AWS_REGION"])
        self.assertEqual("env", session.get_credentials().method)
        self.assertEqual("us-east-1", region)
        self.assertNotIn("never-print-this", stderr.getvalue())
        self.assertIn("AWS_REGION", stderr.getvalue())
        self.assertIn("AWS_ACCESS_KEY_ID", stderr.getvalue())
        self.assertIn("AWS_SECRET_ACCESS_KEY", secret_input.call_args_list[0].args[0])
        self.assertIn("AWS_SESSION_TOKEN", secret_input.call_args_list[1].args[0])
        self.assertEqual("", stdout.getvalue())

    def test_rejected_profile_prompts_directly_for_access_key(self):
        with (
            patch.dict(os.environ, {"AWS_REGION": "us-east-1", "AWS_PROFILE": "broken"}, clear=True),
            patch.object(release, "_boto3", return_value=(self.Boto3, Exception)),
            patch.object(release, "interactive_wizard_enabled", return_value=True),
            patch("builtins.input", side_effect=["AKIADIRECT"]) as user_input,
            patch.object(release.getpass, "getpass", side_effect=["direct-secret", ""]),
            redirect_stderr(StringIO()) as stderr,
        ):
            session, _, *_ = release.session_clients()
            self.assertNotIn("AWS_PROFILE", os.environ)
        self.assertEqual("env", session.get_credentials().method)
        self.assertEqual(1, user_input.call_count)
        self.assertIn("AWS_PROFILE", stderr.getvalue())
        self.assertIn("AWS_ACCESS_KEY_ID", stderr.getvalue())

    def test_rejected_entered_credentials_exit_instead_of_looping(self):
        with (
            patch.dict(os.environ, {"AWS_REGION": "us-east-1", "AWS_PROFILE": "broken"}, clear=True),
            patch.object(release, "_boto3", return_value=(self.Boto3, Exception)),
            patch.object(release, "interactive_wizard_enabled", return_value=True),
            patch.object(
                release, "aws_credential_problem",
                return_value="credential validation returned InvalidClientTokenId",
            ),
            patch("builtins.input", return_value="AKIAREJECTED") as user_input,
            patch.object(release.getpass, "getpass", side_effect=["rejected-secret", ""]),
            redirect_stderr(StringIO()),
            self.assertRaisesRegex(SystemExit, "AWS rejected the values entered"),
        ):
            release.session_clients()
        self.assertEqual(1, user_input.call_count)

    def test_configure_and_no_wizard_options_parse(self):
        args = release.parser().parse_args(["--no-wizard", "configure"])
        self.assertEqual("configure", args.command)
        self.assertTrue(args.no_wizard)


if __name__ == "__main__":
    unittest.main()
