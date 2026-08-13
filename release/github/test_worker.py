#!/usr/bin/env python3

import importlib.util
import os
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "release/github/prepare-worker.py"
SPEC = importlib.util.spec_from_file_location("prepare_worker", MODULE_PATH)
prepare_worker = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(prepare_worker)


class WorkflowMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = prepare_worker.load_json(ROOT / "release/aws/release-plan.json")
        cls.matrix = prepare_worker.load_json(ROOT / "release/github/workflow-matrix.json")

    def test_every_covered_release_workflow_has_exactly_one_worker_mapping(self):
        self.assertEqual(
            set(self.plan["coveredWorkflows"]),
            set(self.matrix["workflows"]),
        )

    def test_every_matrix_row_is_an_explicit_plan_variant(self):
        plan_shards = prepare_worker.plan_shards(self.plan)
        for workflow in self.plan["coveredWorkflows"]:
            rows = (
                prepare_worker.workflow_rows(self.plan, self.matrix, workflow, "linux")
                + prepare_worker.workflow_rows(self.plan, self.matrix, workflow, "host")
            )
            self.assertTrue(rows, workflow)
            self.assertEqual(len(rows), len({row["name"] for row in rows}), workflow)
            for row in rows:
                variants = {
                    variant["name"]
                    for variant in plan_shards[row["shard"]]["build"]["variants"]
                }
                self.assertIn(row["variant"], variants, row["name"])

    def test_linux_cpu_preserves_all_nine_workflow_classifiers(self):
        rows = prepare_worker.workflow_rows(
            self.plan, self.matrix, "build-deploy-linux-x86_64.yml", "linux"
        )
        self.assertEqual(
            [
                "base", "avx2", "avx512", "onednn", "onednn-avx2",
                "onednn-avx512", "compile", "compile-avx2", "compile-avx512",
            ],
            [row["variant"] for row in rows],
        )

    def test_windows_cpu_does_not_invent_managed_llvm_isa_variants(self):
        rows = prepare_worker.workflow_rows(
            self.plan, self.matrix, "build-deploy-windows.yml", "host"
        )
        self.assertEqual(
            [
                "base", "avx2", "avx512", "onednn", "onednn-avx2",
                "onednn-avx512", "compile",
            ],
            [row["variant"] for row in rows],
        )
        self.assertNotIn("compile-avx2", {row["variant"] for row in rows})
        self.assertNotIn("compile-avx512", {row["variant"] for row in rows})

    def test_android_arm64_workflow_includes_cpu_and_vulkan_shards(self):
        rows = prepare_worker.workflow_rows(
            self.plan, self.matrix, "build-deploy-android-arm64.yml", "linux"
        )
        self.assertEqual(
            {"android-arm64", "android-arm64-vulkan"},
            {row["shard"] for row in rows},
        )

    def test_linux_compile_isa_rows_emit_distinct_classifiers(self):
        script = ROOT / "build-scripts/release/linux-x86_64.sh"
        for extension in ("avx2", "avx512"):
            env = os.environ.copy()
            env.update(
                {
                    "DL4J_BUILD_THREADS": "2",
                    "DL4J_EXTENSION": extension,
                    "DL4J_HELPER": "compile",
                    "DL4J_MAVEN_GOAL": "install",
                }
            )
            result = subprocess.run(
                ["bash", str(script), "--print"],
                cwd=ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn(
                f"-Dlibnd4j.classifier=linux-x86_64-compile-{extension}",
                result.stdout,
            )
            self.assertIn(
                f"-Djavacpp.platform.extension=-compile-{extension}",
                result.stdout,
            )

    def test_windows_cuda_bootstrap_preserves_required_cusparse_redists(self):
        bootstrap = (ROOT / "release/github/bootstrap-worker.ps1").read_text()
        self.assertIn("1bd33888dea7d372de612ec9ecc87343ec8dba4a", bootstrap)
        self.assertIn("12.5.4.2", bootstrap)
        self.assertIn("12.5.10.65", bootstrap)
        self.assertIn("cusparse_v2.h", bootstrap)

    def test_unix_protoc_bootstrap_places_member_selector_before_destination(self):
        bootstrap = (ROOT / "release/github/bootstrap-worker.sh").read_text()
        self.assertIn(
            'unzip -qo "${work}/protoc.zip" bin/protoc -d "${toolchain_root}/protoc-21.7"',
            bootstrap,
        )
        self.assertIn('protoc-21.7/bin/protoc" --version', bootstrap)

    def test_optional_android_ndk_bootstrap_returns_successfully(self):
        bootstrap = (ROOT / "release/github/bootstrap-worker.sh").read_text()
        self.assertIn('[ -n "${ndk_version}" ] || return 0', bootstrap)
        self.assertNotIn('[ -n "${ndk_version}" ] || return\n', bootstrap)

    def test_native_launcher_expands_empty_arrays_safely_on_macos_bash(self):
        launcher = (ROOT / "build-scripts/release/native-platform.sh").read_text()
        self.assertIn('${split_flags[@]+"${split_flags[@]}"}', launcher)
        self.assertIn('${repo[@]+"${repo[@]}"}', launcher)
        self.assertIn('${win[@]+"${win[@]}"}', launcher)
        self.assertIn('${zluda_win[@]+"${zluda_win[@]}"}', launcher)


class WorkerConfigTests(unittest.TestCase):
    def args(self, **overrides):
        values = {
            "plan": ROOT / "release/aws/release-plan.json",
            "source": ROOT,
            "shard": "linux-x86_64-cpu",
            "variant": "compile-avx2",
            "build_threads": "8",
            "maven_flags": "-Dexample=true",
            "libnd4j_url": "",
            "build_aot": False,
            "aot_all_spins": False,
            "azure_cache": True,
            "release_version": "1.0.0-SNAPSHOT",
            "snapshot_version": "1.0.0-SNAPSHOT",
            "run_id": "gha-test",
            "commit": "abc123",
        }
        values.update(overrides)
        return type("Args", (), values)()

    def test_config_selects_one_variant_and_references_secret_by_name(self):
        config = prepare_worker.worker_config(self.args())
        self.assertEqual(
            ["compile-avx2"],
            [variant["name"] for variant in config["shard"]["build"]["variants"]],
        )
        self.assertEqual(8, config["shard"]["build"]["buildThreads"])
        self.assertEqual("-Dexample=true", config["shard"]["build"]["workflowMvnFlags"])
        self.assertEqual("1.0.0-SNAPSHOT", config["snapshotVersion"])
        self.assertEqual(
            "SCCACHE_AZURE_CONNECTION_STRING",
            config["compilerCache"]["connectionStringEnv"],
        )
        self.assertNotIn("connectionString", config["compilerCache"])

    def test_default_threads_are_capped_to_the_runner_cpu_count(self):
        with patch.object(prepare_worker.os, "cpu_count", return_value=4):
            config = prepare_worker.worker_config(self.args(build_threads=""))
        self.assertEqual(4, config["shard"]["build"]["buildThreads"])

    def test_aot_defaults_to_base_unless_all_spins_is_enabled(self):
        self.assertFalse(
            prepare_worker.worker_config(self.args(build_aot=True))["shard"]["build"]["buildAot"]
        )
        self.assertTrue(
            prepare_worker.worker_config(
                self.args(build_aot=True, aot_all_spins=True)
            )["shard"]["build"]["buildAot"]
        )


if __name__ == "__main__":
    unittest.main()
