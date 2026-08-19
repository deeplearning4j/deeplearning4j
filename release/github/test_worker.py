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

    def test_existing_dispatch_workflow_can_select_branch_only_matrices(self):
        workflow = (ROOT / ".github/workflows/build-deploy-cross-platform.yml").read_text()
        self.assertIn("workflow:\n        description: Canonical release workflow matrix to execute.", workflow)
        self.assertIn("workflow: ${{ inputs.workflow }}", workflow)

    def test_reusable_worker_resolves_abbreviated_source_refs_once(self):
        workflow = (ROOT / ".github/workflows/_release-worker.yml").read_text()
        self.assertIn("const ref = process.env.SOURCE_REF || context.sha;", workflow)
        self.assertIn("github.rest.repos.getCommit", workflow)
        self.assertIn("source: ${{ steps.source.outputs.result }}", workflow)
        self.assertEqual(
            1, workflow.count("ref: ${{ steps.source.outputs.result }}")
        )
        self.assertEqual(
            4, workflow.count("ref: ${{ needs.matrix.outputs.source }}")
        )
        self.assertNotIn("ref: ${{ inputs.sourceRef }}", workflow)

    def test_every_dispatch_caller_requires_explicit_targeted_retry_mode(self):
        for workflow_name in self.matrix["workflows"]:
            workflow = (ROOT / ".github/workflows" / workflow_name).read_text()
            self.assertIn("      classifiers:\n", workflow, workflow_name)
            self.assertIn("      targetedRetry:\n", workflow, workflow_name)
            self.assertIn(
                "      targetedRetry: ${{ inputs.targetedRetry }}",
                workflow,
                workflow_name,
            )

        reusable = (ROOT / ".github/workflows/_release-worker.yml").read_text()
        self.assertIn("          TARGETED_RETRY: ${{ inputs.targetedRetry }}", reusable)
        self.assertEqual(2, reusable.count('--selection-mode "$selection_mode"'))

    def test_every_covered_release_workflow_has_exactly_one_worker_mapping(self):
        self.assertEqual(
            set(self.plan["coveredWorkflows"]),
            set(self.matrix["workflows"]),
        )

    def test_every_plan_shard_is_reachable_from_a_workflow(self):
        plan_shards = set(prepare_worker.plan_shards(self.plan))
        mapped_shards = {
            row["shard"]
            for rows in self.matrix["workflows"].values()
            for row in rows
        }
        self.assertEqual(plan_shards, mapped_shards)

    def test_every_plan_variant_is_reachable_from_a_canonical_workflow(self):
        plan_shards = prepare_worker.plan_shards(self.plan)
        expected = {
            prepare_worker.public_artifact_id(
                shard_id,
                prepare_worker.canonical_variant_name(shard, str(variant["name"])),
                variant,
            )
            for shard_id, shard in plan_shards.items()
            for variant in shard["build"].get("variants", [])
        }
        actual = set()
        for workflow in self.plan["coveredWorkflows"]:
            rows = (
                prepare_worker.workflow_rows(self.plan, self.matrix, workflow, "linux")
                + prepare_worker.workflow_rows(self.plan, self.matrix, workflow, "host")
            )
            actual.update(row["selector"] for row in rows)
        self.assertEqual(expected, actual)

    def test_public_artifact_id_does_not_duplicate_shard_suffix(self):
        self.assertEqual(
            "linux-x86_64-compat",
            prepare_worker.public_artifact_id("linux-x86_64-compat", "compat"),
        )
        self.assertEqual(
            "linux-x86_64",
            prepare_worker.public_artifact_id("linux-x86_64-cpu", "base"),
        )
        self.assertEqual(
            "linux-x86_64-avx2",
            prepare_worker.public_artifact_id("linux-x86_64-cpu", "avx2"),
        )
        self.assertEqual(
            "linux-x86_64-cuda-12-9",
            prepare_worker.public_artifact_id(
                "linux-x86_64-cuda-12-9", "base", {"classifierSuffix": "-cuda-12.9"}
            ),
        )
        self.assertEqual(
            "linux-x86_64-cuda-12.9-zluda-rocm-6.2.4",
            prepare_worker.public_artifact_id(
                "linux-x86_64-zluda-rocm-6.2.4",
                "cuda-12.9",
                {"classifierSuffix": "-cuda-12.9-zluda-rocm-6.2.4"},
            ),
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
            self.assertEqual(len(rows), len({row["artifactId"] for row in rows}), workflow)
            for row in rows:
                self.assertNotIn("--", row["name"], workflow)
                self.assertNotIn("--", row["artifactId"], workflow)
                self.assertNotIn("zluda-zluda", row["name"], workflow)
                self.assertNotIn("zluda-zluda", row["artifactId"], workflow)
                self.assertNotRegex(row["artifactId"], r"-cpu(?:-|$)", workflow)
                if "-zluda" in row["shard"]:
                    self.assertRegex(
                        row["artifactId"], r"-rocm-[0-9]+\.[0-9]+\.[0-9]+$", workflow
                    )
                self.assertEqual(row["name"], row["artifactId"], workflow)
                self.assertTrue(
                    row["dependencyCacheKey"].startswith(f'{row["shard"]}-'), workflow
                )
                self.assertEqual(row["artifactId"], row["selector"], workflow)
                self.assertNotIn("--", row["selector"], workflow)
                if row["variant"] == "base":
                    self.assertEqual(
                        prepare_worker.public_artifact_id(row["shard"], "base"),
                        row["artifactId"],
                        workflow,
                    )
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

    def test_dependency_caches_are_scoped_by_variant_toolchain_contract(self):
        linux_cpu = prepare_worker.workflow_rows(
            self.plan, self.matrix, "build-deploy-linux-x86_64.yml", "linux"
        )
        linux_scopes = {row["variant"]: row["dependencyCacheKey"] for row in linux_cpu}
        self.assertEqual("linux-x86_64-cpu-default", linux_scopes["base"])
        self.assertEqual(linux_scopes["base"], linux_scopes["avx2"])
        self.assertEqual(linux_scopes["onednn"], linux_scopes["onednn-avx512"])
        self.assertIn("-onednn-", linux_scopes["onednn"])
        self.assertEqual(linux_scopes["compile"], linux_scopes["compile-avx512"])
        self.assertIn("-mlir-", linux_scopes["compile"])
        self.assertNotEqual(linux_scopes["base"], linux_scopes["compile"])

        linux_cuda = prepare_worker.workflow_rows(
            self.plan, self.matrix, "build-deploy-linux-cuda-12.9.yml", "linux"
        )
        cuda_scopes = {row["variant"]: row["dependencyCacheKey"] for row in linux_cuda}
        self.assertEqual("linux-x86_64-cuda-12-9-default", cuda_scopes["base"])
        self.assertIn("-cudnn-", cuda_scopes["cudnn"])
        self.assertIn("-triton-", cuda_scopes["compile"])
        self.assertEqual(3, len(set(cuda_scopes.values())))

        windows_cpu = prepare_worker.workflow_rows(
            self.plan, self.matrix, "build-deploy-windows.yml", "host"
        )
        windows_scopes = {
            row["variant"]: row["dependencyCacheKey"] for row in windows_cpu
        }
        self.assertIn("-windows-native-compile-", windows_scopes["compile"])
        self.assertNotEqual(windows_scopes["base"], windows_scopes["compile"])

    def test_android_arm64_workflow_includes_cpu_and_vulkan_shards(self):
        rows = prepare_worker.workflow_rows(
            self.plan, self.matrix, "build-deploy-android-arm64.yml", "linux"
        )
        self.assertEqual(
            {"android-arm64", "android-arm64-vulkan"},
            {row["shard"] for row in rows},
        )
        compile_rows = [row for row in rows if row["variant"] == "compile-nnapi"]
        self.assertTrue(compile_rows)
        self.assertEqual(1, len({row["dependencyCacheKey"] for row in compile_rows}))
        self.assertIn("-mlir-no-triton-", compile_rows[0]["dependencyCacheKey"])

    def test_exact_classifier_filter_selects_only_requested_row(self):
        rows = prepare_worker.workflow_rows(
            self.plan,
            self.matrix,
            "build-deploy-android-arm64.yml",
            "linux",
            classifiers="android-arm64-vulkan",
            selection_mode="targeted",
        )
        self.assertEqual(
            [("android-arm64-vulkan", "base")],
            [(row["shard"], row["variant"]) for row in rows],
        )
        self.assertEqual("android-arm64-vulkan", rows[0]["artifactId"])
        self.assertEqual("android-arm64-vulkan", rows[0]["selector"])

    def test_targeted_classifier_auto_resolves_unique_canonical_workflow(self):
        rows = prepare_worker.workflow_rows(
            self.plan,
            self.matrix,
            "build-deploy-cross-platform.yml",
            "host",
            classifiers="windows-x86_64-cuda-12.9-zluda-rocm-7.2.4",
            selection_mode="targeted",
        )
        self.assertEqual(
            [("windows-x86_64-zluda", "cuda-12.9")],
            [(row["shard"], row["variant"]) for row in rows],
        )
        self.assertEqual(
            "windows-x86_64-cuda-12.9-zluda-rocm-7.2.4",
            rows[0]["artifactId"],
        )
        self.assertNotIn("--", rows[0]["artifactId"])
        self.assertNotIn("zluda-zluda", rows[0]["artifactId"])
        self.assertIn("-rocm-7.2.4", rows[0]["artifactId"])
        self.assertEqual(rows[0]["artifactId"], rows[0]["selector"])

    def test_public_classifier_id_resolves_to_public_selector(self):
        rows = prepare_worker.workflow_rows(
            self.plan,
            self.matrix,
            "build-deploy-windows.yml",
            "host",
            classifiers="windows-x86_64-avx2",
            selection_mode="targeted",
        )
        self.assertEqual(
            [("windows-x86_64-cpu", "avx2")],
            [(row["shard"], row["variant"]) for row in rows],
        )
        self.assertEqual("windows-x86_64-avx2", rows[0]["selector"])
        self.assertNotIn("--", rows[0]["selector"])

    def test_public_zluda_classifier_id_auto_resolves_canonical_workflow(self):
        rows = prepare_worker.workflow_rows(
            self.plan,
            self.matrix,
            "build-deploy-cross-platform.yml",
            "host",
            classifiers="windows-x86_64-cuda-12.9-zluda-rocm-7.2.4",
            selection_mode="targeted",
        )
        self.assertEqual(
            [("windows-x86_64-zluda", "cuda-12.9")],
            [(row["shard"], row["variant"]) for row in rows],
        )
        self.assertEqual(
            "windows-x86_64-cuda-12.9-zluda-rocm-7.2.4",
            rows[0]["artifactId"],
        )

    def test_classifier_filter_rejects_unknown_row(self):
        with self.assertRaisesRegex(ValueError, "does not contain requested classifiers"):
            prepare_worker.workflow_rows(
                self.plan,
                self.matrix,
                "build-deploy-android-arm64.yml",
                "linux",
                classifiers="android-arm64-vulkan-missing",
                selection_mode="targeted",
            )

    def test_complete_matrix_rejects_classifier_filters(self):
        with self.assertRaisesRegex(ValueError, "selection_mode='targeted'"):
            prepare_worker.workflow_rows(
                self.plan,
                self.matrix,
                "build-deploy-windows.yml",
                "host",
                classifiers="windows-x86_64-avx2",
            )

    def test_targeted_matrix_requires_explicit_classifiers(self):
        with self.assertRaisesRegex(ValueError, "requires at least one classifier"):
            prepare_worker.workflow_rows(
                self.plan,
                self.matrix,
                "build-deploy-windows.yml",
                "host",
                selection_mode="targeted",
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
        self.assertIn("Windows.flatc.binary.zip", bootstrap)
        self.assertIn("DL4J_FLATC_EXECUTABLE", bootstrap)
        self.assertIn("VC\\Auxiliary\\Build\\vcvars64.bat", bootstrap)
        self.assertIn("Microsoft.VisualStudio.Component.VC.Tools.x86.x64", bootstrap)
        self.assertIn("Get-Command cl.exe", bootstrap)
        self.assertIn("Add-Content -Path $env:GITHUB_PATH", bootstrap)
        self.assertIn("$originalPathSet.Contains($pathEntry)", bootstrap)
        self.assertNotIn("if ($pathEntry) {\n      Add-Content -Path $env:GITHUB_PATH", bootstrap)
        self.assertIn("'INCLUDE'", bootstrap)
        self.assertIn("'LIB'", bootstrap)

    def test_macos_mlx_sources_match_the_pinned_mlx_api(self):
        builder = (ROOT / "libnd4j/include/graph/cpu/MlxIRBuilder.cpp").read_text()
        header = (ROOT / "libnd4j/include/graph/cpu/MlxIRBuilder.h").read_text()
        platform = (ROOT / "libnd4j/cmake/Platform.cmake").read_text()
        detection = (ROOT / "libnd4j/cmake/PlatformDetection.cmake").read_text()
        types = (ROOT / "libnd4j/include/types/types.h").read_text()
        mps_dir = ROOT / "libnd4j/include/ops/declarable/platform/mps"
        mps_header = (mps_dir / "mpsUtils.h").read_text()
        mps_sources = "\n".join(path.read_text() for path in mps_dir.glob("*.mm"))
        self.assertIn("struct Dtype;", header)
        self.assertIn("GraphAnalysisUtils::profileSegment", builder)
        self.assertIn("std::optional<mx::array> mask", builder)
        self.assertIn("std::nullopt", builder)
        self.assertNotIn("OpCategoryTable::categorize", builder)
        self.assertNotIn("mx::array()", builder)
        self.assertNotIn("mmacosx-version-min=10.10", platform)
        self.assertNotIn("mmacosx-version-min=10.10", detection)
        self.assertIn('CACHE STRING "Minimum macOS deployment target" FORCE', platform)
        self.assertIn('CACHE STRING "Minimum macOS deployment target" FORCE', detection)
        self.assertIn("!defined(_WIN32) && !defined(__OBJC__)", types)
        self.assertIn("static constexpr auto BOOL = sd::DataType::BOOL;", types)
        self.assertLess(
            mps_header.index("#import <Foundation/Foundation.h>"),
            mps_header.index("#import <Metal/Metal.h>"),
        )
        self.assertNotIn("const sd::NDArray", mps_header)
        self.assertNotIn("const sd::NDArray", mps_sources)
        self.assertNotIn("const NDArray", mps_sources)

    def test_compat_worker_uses_modern_container_python(self):
        action = (ROOT / ".github/actions/run-release-worker/action.yml").read_text()
        bootstrap = (ROOT / "release/github/bootstrap-worker.sh").read_text()
        self.assertIn("inputs.shard != 'linux-x86_64-compat'", action)
        self.assertIn("python3.11 python3.10 python python3", action)
        self.assertIn("sys.version_info < (3, 10)", action)
        self.assertIn("python3 python3.11", bootstrap)

    def test_worker_stages_only_release_outputs_and_prunes_android_vulkan_intermediates(self):
        action = (ROOT / ".github/actions/run-release-worker/action.yml").read_text()
        workflow = (ROOT / ".github/workflows/_release-worker.yml").read_text()
        vulkan_pom = (
            ROOT
            / "nd4j/nd4j-backends/nd4j-backend-impls/nd4j-vulkan/pom.xml"
        ).read_text()

        self.assertIn(
            'work_root="${RUNNER_TEMP}/dl4j-release-work/${INPUT_SHARD}/${INPUT_VARIANT}"',
            action,
        )
        self.assertIn('--repository "${work_root}/m2"', action)
        self.assertNotIn('--repository "${artifact_root}/m2"', action)
        self.assertEqual(2, workflow.count("compression-level: 0"))
        self.assertIn(
            '<target if="dl4j.prune.native.intermediates">', vulkan_pom
        )
        self.assertIn('<include name="**/*.o"/>', vulkan_pom)
        self.assertIn('<include name="**/*.a"/>', vulkan_pom)
        self.assertIn("<value>android-arm64</value>", vulkan_pom)
        self.assertIn("<value>android-x86_64</value>", vulkan_pom)

    def test_every_worker_reuses_native_and_maven_dependency_downloads(self):
        action = (ROOT / ".github/actions/run-release-worker/action.yml").read_text()
        workflow = (ROOT / ".github/workflows/_release-worker.yml").read_text()
        preparer = (ROOT / "release/github/prepare-worker.py").read_text()

        self.assertIn("name: Restore Maven dependency repository", action)
        self.assertIn(
            "${{ runner.temp }}/dl4j-release-work/"
            "${{ inputs.shard }}/${{ inputs.variant }}/m2",
            action,
        )
        self.assertIn("dl4j-maven-v2-${{ runner.os }}-", action)
        self.assertIn("name: Keep only third-party Maven dependencies in cache", action)
        self.assertIn(
            '"${maven_cache}/org/eclipse/deeplearning4j"', action
        )
        self.assertEqual(
            2, workflow.count("name: Restore managed dependency downloads")
        )
        self.assertEqual(4, workflow.count("dl4j-dependency-v4-${{ runner.os }}-"))
        self.assertNotIn("if: matrix.dependencyCacheKey != ''", workflow)
        self.assertIn("DL4J_DEPENDENCY_CACHE_HELPER: ${{ github.workspace }}/release/azure/dependency-cache.py", action)
        self.assertIn("DL4J_CLOUD_IO: ${{ github.workspace }}/release/azure/cloud-io.py", action)
        self.assertIn("toolchainCache", preparer)
        self.assertIn("deeplearning4j/releases/toolchain-cache/v1", preparer)

    def test_shared_worker_publishes_prebuilt_snapshots_with_existing_central_credentials(self):
        action = (ROOT / ".github/actions/run-release-worker/action.yml").read_text()
        workflow = (ROOT / ".github/workflows/_release-worker.yml").read_text()
        preparer = (ROOT / "release/github/prepare-worker.py").read_text()

        self.assertIn("release-version:", action)
        self.assertIn("python-executable:", action)
        self.assertIn(
            'print(json.load(open(sys.argv[1], encoding="utf-8"))["releaseVersion"])',
            action,
        )
        self.assertIn("worker-success", action)
        self.assertIn('subparsers.add_parser("release-version")', preparer)
        self.assertIn("CENTRAL_SONATYPE_TOKEN_USERNAME:\n        required: true", workflow)
        self.assertIn("CENTRAL_SONATYPE_TOKEN_PASSWORD:\n        required: true", workflow)
        self.assertEqual(2, workflow.count("name: Download staged worker result"))
        self.assertEqual(2, workflow.count("name: Publish staged Maven snapshot"))
        self.assertEqual(2, workflow.count("repository.py deploy-snapshot"))
        self.assertEqual(2, workflow.count("server-id: central-portal-snapshots"))
        self.assertEqual(2, workflow.count('if [ ! -f "${artifact_root}/worker-success" ]'))
        self.assertEqual(
            2,
            workflow.count(
                "--url https://central.sonatype.com/repository/maven-snapshots/"
            ),
        )
        self.assertIn("needs: [matrix, linux]", workflow)
        self.assertIn("needs: [matrix, host]", workflow)
        self.assertEqual(2, workflow.count("max-parallel: 1"))
        self.assertIn("if: ${{ always() && needs.matrix.result == 'success'", workflow)
        self.assertNotIn("endsWith(steps.worker.outputs['release-version']", workflow)
        for caller in sorted((ROOT / ".github/workflows").glob("build-deploy-*")):
            contents = caller.read_text()
            if "uses: ./.github/workflows/_release-worker.yml" in contents:
                self.assertIn("secrets: inherit", contents, caller.name)

    def test_staged_worker_artifacts_can_be_republished_without_rebuilding(self):
        workflow = (
            ROOT / ".github/workflows/publish-release-worker-artifacts.yml"
        ).read_text()

        self.assertIn("sourceRunId:", workflow)
        self.assertIn("artifactPattern:", workflow)
        self.assertIn("listWorkflowRunArtifacts", workflow)
        self.assertIn("listJobsForWorkflowRun", workflow)
        self.assertIn("workerStep?.conclusion !== 'success'", workflow)
        self.assertIn("uploadStep?.conclusion !== 'success'", workflow)
        self.assertIn("actions/download-artifact@v8", workflow)
        self.assertIn("run-id: ${{ inputs.sourceRunId }}", workflow)
        self.assertIn("max-parallel: 1", workflow)
        self.assertIn('if [ ! -f "${artifact_root}/worker-config.json" ]', workflow)
        self.assertNotIn('if [ ! -f "${artifact_root}/worker-success" ]', workflow)
        self.assertIn("repository.py deploy-snapshot", workflow)
        self.assertNotIn("run-release-worker", workflow)

        caller = (
            ROOT / ".github/workflows/build-deploy-cross-platform.yml"
        ).read_text()
        self.assertIn("publishSourceRunId:", caller)
        self.assertIn("permissions:\n  actions: read\n  contents: read", caller)
        self.assertIn("if: inputs.publishSourceRunId == ''", caller)
        self.assertIn("if: inputs.publishSourceRunId != ''", caller)
        self.assertIn(
            "uses: ./.github/workflows/publish-release-worker-artifacts.yml",
            caller,
        )

    def test_github_actions_use_current_node24_runtimes(self):
        github_files = sorted((ROOT / ".github").rglob("*.yml"))
        github_files.extend(sorted((ROOT / ".github").rglob("*.yaml")))
        contents = {path: path.read_text() for path in github_files}
        deprecated = (
            "actions/setup-java@v4",
            "actions/download-artifact@v4",
            "actions/upload-artifact@v4",
            "actions/cache@v4",
            "actions/github-script@v7",
        )
        for path, text in contents.items():
            for action in deprecated:
                self.assertNotIn(action, text, path)
        combined = "\n".join(contents.values())
        for action in (
            "actions/setup-java@v5",
            "actions/download-artifact@v8",
            "actions/upload-artifact@v7",
            "actions/cache@v6",
            "actions/github-script@v9",
        ):
            self.assertIn(action, combined)

    def test_linux_bootstrap_pins_a_supported_maven(self):
        bootstrap = (ROOT / "release/github/bootstrap-worker.sh").read_text()
        self.assertIn("ensure_modern_maven()", bootstrap)
        self.assertIn("maven_version=3.9.9", bootstrap)
        self.assertIn('"${target}/bin/mvn" --version', bootstrap)
        self.assertIn('"${target}/bin" >>"${GITHUB_PATH}"', bootstrap)

    def test_android_worker_accepts_sccache_as_the_required_compiler_cache(self):
        builder = (ROOT / "libnd4j/buildnativeoperations.sh").read_text()
        self.assertIn("ccache|ccache.exe)", builder)
        self.assertIn("sccache|sccache.exe)", builder)
        self.assertIn("-DSCCACHE_PROGRAM:FILEPATH=$DL4J_COMPILER_CACHE", builder)
        self.assertNotIn(
            "Android smart-cache contract requires ccache",
            builder,
        )

    def test_external_dependencies_receive_complete_android_and_host_tool_contracts(self):
        dependencies = (ROOT / "libnd4j/cmake/Dependencies.cmake").read_text()
        self.assertIn("ONEDNN_CMAKE_ARGS -DANDROID_ABI=", dependencies)
        self.assertIn("ONEDNN_CMAKE_ARGS -DANDROID_PLATFORM=", dependencies)
        self.assertIn("DL4J_FLATC_EXECUTABLE", dependencies)

    def test_sccache_actions_support_github_and_azure_backends(self):
        for operating_system in ("linux", "windows", "macos"):
            action = (
                ROOT / f".github/actions/setup-sccache-{operating_system}/action.yml"
            ).read_text()
            self.assertIn("--features gha,azure --no-default-features", action)
            self.assertIn("gha-azure", action)

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

    def test_cross_platform_launcher_expands_empty_arrays_safely_on_macos_bash(self):
        launcher = (ROOT / "build-scripts/release/cross-platform.sh").read_text()
        self.assertIn('${mingw[@]+"${mingw[@]}"}', launcher)
        self.assertIn('${repository[@]+"${repository[@]}"}', launcher)
        self.assertIn('${protoc_profile[@]+"${protoc_profile[@]}"}', launcher)

    def test_windows_tokenizers_builds_rust_for_mingw(self):
        builder = (
            ROOT / "nd4j/nd4j-tokenizers/libtokenizers/buildnativetokenizers.sh"
        ).read_text()
        self.assertIn('CARGO_BUILD_TARGET="${TARGET}"', builder)
        self.assertIn('RUST_TOOLCHAIN="stable-${TARGET}"', builder)
        self.assertIn('DL4J_MINGW_BIN', builder)
        self.assertIn('gcc -dumpmachine', builder)
        self.assertIn('--target "${CARGO_BUILD_TARGET}"', builder)

    def test_windows_worker_pins_one_mingw_toolchain(self):
        action = (ROOT / ".github/actions/run-release-worker/action.yml").read_text()
        bootstrap = (ROOT / "release/github/bootstrap-worker.ps1").read_text()
        self.assertIn("msys2/setup-msys2@v2", action)
        self.assertIn("msystem: MINGW64", action)
        self.assertIn("release: false", action)
        self.assertIn("DL4J_MINGW_BIN", bootstrap)
        self.assertIn("x86_64-w64-mingw32", bootstrap)

    def test_cuda_linking_includes_cublas_lt(self):
        configuration = (ROOT / "libnd4j/cmake/CudaConfiguration.cmake").read_text()
        sdx_configuration = (ROOT / "libnd4j/cmake/BuildSDX.cmake").read_text()
        self.assertIn(
            "CUDA::cublas CUDA::cublasLt CUDA::cusolver",
            configuration,
        )
        self.assertIn(
            "CUDA::cudart CUDA::cublas CUDA::cublasLt CUDA::cusolver",
            sdx_configuration,
        )
        self.assertIn('target_link_options(${main_target_name} PRIVATE "LINKER:--no-as-needed")', configuration)
        self.assertIn('target_link_options(${main_target_name} PRIVATE "LINKER:--no-as-needed")', sdx_configuration)

    def test_zluda_profile_skips_only_the_duplicate_cuda_assembly(self):
        pom = (ROOT / "libnd4j/pom.xml").read_text()
        self.assertIn(
            "<libnd4j.cuda.assembly.skip>false</libnd4j.cuda.assembly.skip>",
            pom,
        )
        self.assertIn(
            "<skipAssembly>${libnd4j.cuda.assembly.skip}</skipAssembly>",
            pom,
        )
        self.assertIn(
            "<id>zluda</id>",
            pom,
        )
        self.assertIn(
            "<libnd4j.cuda.assembly.skip>true</libnd4j.cuda.assembly.skip>",
            pom,
        )
        self.assertIn("<id>zluda-msvc</id>", pom)
        self.assertIn("<family>windows</family>", pom)
        self.assertIn("<name>libnd4j.zluda</name>", pom)
        self.assertIn("<compilerOption>/MT</compilerOption>", pom)

        api_pom = (
            ROOT
            / "nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/pom.xml"
        ).read_text()
        self.assertIn("<classifier>${libnd4j.classifier}</classifier>", api_pom)
        self.assertNotIn(
            "<classifier>${javacpp.platform}-cuda-${cuda.version}</classifier>",
            api_pom,
        )

    def test_zluda_uses_shared_cuda_runtime_archives(self):
        configuration = (ROOT / "libnd4j/cmake/CudaConfiguration.cmake").read_text()
        self.assertIn(
            "function(link_zluda_cuda_shared_library main_target_name imported_target library_name required)",
            configuration,
        )
        self.assertNotIn("link_zluda_cuda_static_library", configuration)
        self.assertNotIn("CUDA::cudart_static", configuration)
        self.assertNotIn("CUDA::nvrtc_static", configuration)
        self.assertNotIn("CUDA::cusolver_static", configuration)
        self.assertNotIn("cusolver_lapack_static", configuration)
        self.assertIn(
            "${main_target_name} CUDA::cudart cudart TRUE",
            configuration,
        )
        self.assertIn(
            "${main_target_name} CUDA::nvrtc nvrtc TRUE",
            configuration,
        )
        self.assertIn(
            '"ZLUDA build requires ${imported_target} (${library_name}); install the CUDA runtime toolkit"',
            configuration,
        )
        self.assertIn(
            'if(WIN32)\n        find_library(_zluda_cuda_shared_library',
            configuration,
        )
        self.assertIn(
            'PATH_SUFFIXES\n                lib/x64 lib64 lib',
            configuration,
        )
        self.assertIn(
            'CUDA::cuda_driver nvcuda TRUE',
            configuration,
        )
        self.assertIn(
            'CUDA_RESOLVE_DEVICE_SYMBOLS OFF',
            (ROOT / "libnd4j/cmake/MainBuildFlow.cmake").read_text(),
        )
        self.assertIn(
            'CUDA_RUNTIME_LIBRARY Shared',
            (ROOT / "libnd4j/cmake/MainBuildFlow.cmake").read_text(),
        )
        self.assertIn(
            'ZLUDA supports compute_80, compute_86, and compute_89',
            configuration,
        )
        self.assertNotIn('set(_requested "8.6 9.0")', configuration)

        native_platform = (
            ROOT / "build-scripts/release/native-platform.sh"
        ).read_text()
        zluda_block = native_platform.split("  zluda|windows-zluda)", 1)[1].split(
            "    ;;", 1
        )[0]
        self.assertNotIn("-Dlibnd4j.compute=8.6 9.0", zluda_block)
        self.assertIn(
            "Let CMake resolve the ZLUDA virtual architecture",
            zluda_block,
        )

        zluda_configuration = (
            ROOT / "libnd4j/cmake/ZludaConfiguration.cmake"
        ).read_text()
        self.assertIn(
            'set(CMAKE_MSVC_RUNTIME_LIBRARY\n            "MultiThreaded$<$<CONFIG:Debug>:Debug>" PARENT_SCOPE)',
            zluda_configuration,
        )
        self.assertIn(
            'Windows ZLUDA MSVC runtime: static (/MT)',
            zluda_configuration,
        )

    def test_zluda_release_plans_leave_ptx_architecture_to_cmake(self):
        for provider in ("aws", "azure", "gcp"):
            plan = prepare_worker.load_json(ROOT / f"release/{provider}/release-plan.json")
            for shard in plan["shards"]:
                build = shard["build"]
                if build.get("zludaVersion"):
                    self.assertNotIn(
                        "-Dlibnd4j.compute=8.6 9.0",
                        build.get("mavenArgs", []),
                        f"{provider}:{shard['id']}",
                    )

        cmake_source = (ROOT / "libnd4j/CMakeLists.txt").read_text()
        self.assertIn(
            'if(SD_ZLUDA)\n        set(_sd_cuda_msvc_runtime_flag "-MT")',
            cmake_source,
        )
        self.assertIn(
            'set(_sd_cuda_msvc_runtime_flag "-MD")',
            cmake_source,
        )
        self.assertIn(
            '-Xcompiler=${_sd_cuda_msvc_runtime_flag}',
            cmake_source,
        )
        self.assertIn(
            'if(SD_ZLUDA)\n            set(_sd_target_msvc_runtime',
            cmake_source,
        )
        self.assertIn(
            'MSVC_RUNTIME_LIBRARY "${_sd_target_msvc_runtime}"',
            cmake_source,
        )
        self.assertIn(
            'CUDA_RESOLVE_DEVICE_SYMBOLS OFF',
            cmake_source,
        )
        self.assertIn(
            'CUDA_SEPARABLE_COMPILATION OFF',
            cmake_source,
        )
        self.assertIn(
            'CUDA_RUNTIME_LIBRARY Shared',
            cmake_source,
        )

        cublas_helper = (
            ROOT / "libnd4j/include/helpers/cuda/cublasHelper.cu"
        ).read_text()
        svd_helper = (
            ROOT / "libnd4j/include/ops/declarable/helpers/cuda/svd.cu"
        ).read_text()
        lup_helper = (
            ROOT / "libnd4j/include/ops/declarable/helpers/cuda/lup.cu"
        ).read_text()
        self.assertIn("#if !defined(HAVE_ZLUDA)", cublas_helper)
        self.assertIn(
            "SVD requires cuSolver and is not supported by the ZLUDA backend",
            svd_helper,
        )
        self.assertIn(
            "LUP factorization requires cuSolver and is not supported by the ZLUDA backend",
            lup_helper,
        )

    def test_native_builder_avoids_bash4_uppercase_expansion(self):
        builder = (ROOT / "libnd4j/buildnativeoperations.sh").read_text()
        self.assertNotIn("^^}", builder)
        self.assertIn("uppercase()", builder)

    def test_macos_mps_sources_use_supported_metal_and_ndarray_apis(self):
        cmake_source = (ROOT / "libnd4j/CMakeLists.txt").read_text()
        sdx_linking = (ROOT / "libnd4j/cmake/BuildSDX.cmake").read_text()
        self.assertIn("find_program(CMAKE_OTOOL NAMES otool)", cmake_source)
        self.assertIn(
            "target_link_libraries(${main_target_name} PUBLIC ${MPS_LIBRARIES})",
            sdx_linking,
        )

        activations = (
            ROOT
            / "libnd4j/include/ops/declarable/platform/mps/mps_activations.mm"
        ).read_text()
        self.assertNotIn("MPSCNNNeuronGeLU", activations)
        self.assertIn("newLibraryWithSource", activations)
        self.assertIn("dispatchThreads", activations)

        blas = (
            ROOT / "libnd4j/include/ops/declarable/platform/mps/mps_blas.mm"
        ).read_text()
        self.assertNotIn(".subarray(", blas)
        self.assertNotIn("reshape(a->ordering(), {", blas)
        self.assertIn("(*aReshape)(i, {0})", blas)

        conv = (
            ROOT / "libnd4j/include/ops/declarable/platform/mps/mps_conv.mm"
        ).read_text()
        self.assertNotIn("MPSOffsetMake", conv)
        self.assertEqual(2, conv.count("MPSOffset poolingOffset ="))

        image = (
            ROOT / "libnd4j/include/ops/declarable/platform/mps/mps_image.mm"
        ).read_text()
        self.assertNotIn("PLATFORM_IMPL(depthwise_conv2d, ENGINE_MPS)", conv)
        self.assertNotIn("PLATFORM_CHECK(depthwise_conv2d, ENGINE_MPS)", conv)
        self.assertEqual(
            2,
            image.count("PLATFORM_IMPL(depthwise_conv2d, ENGINE_MPS)"),
        )
        self.assertEqual(
            2,
            image.count("PLATFORM_CHECK(depthwise_conv2d, ENGINE_MPS)"),
        )

        mps_dir = ROOT / "libnd4j/include/ops/declarable/platform/mps"
        registration_sources = [mps_dir / "mpsUtils.h", *mps_dir.glob("mps_*.mm")]
        for source in registration_sources:
            contents = source.read_text()
            self.assertNotIn("ENGINE_CPU", contents, source)
            self.assertIn("ENGINE_MPS", contents, source)

        elementwise = (
            ROOT / "libnd4j/include/ops/declarable/platform/mps/mps_elementwise.mm"
        ).read_text()
        self.assertNotIn("start:0", elementwise)
        self.assertNotIn("scale:nil", elementwise)
        self.assertIn("scaleVector:nil", elementwise)
        self.assertIn("startIndex:0", elementwise)

    def test_native_build_logs_are_compact_unless_verbose_is_explicit(self):
        setup = (ROOT / "libnd4j/cmake/Setup.cmake").read_text()
        compiler_flags = (ROOT / "libnd4j/cmake/CompilerFlags.cmake").read_text()
        cuda_config = (ROOT / "libnd4j/cmake/CudaConfiguration.cmake").read_text()
        build_script = (ROOT / "libnd4j/buildnativeoperations.sh").read_text()
        nvcc_filter = (ROOT / "libnd4j/cmake/nvcc_filter.py").read_text()

        self.assertIn('option(SD_VERBOSE_BUILD "Print full native compiler command lines" OFF)', setup)
        self.assertNotIn("set(CMAKE_VERBOSE_MAKEFILE ON)", compiler_flags)
        self.assertNotIn("set(CMAKE_VERBOSE_MAKEFILE ON PARENT_SCOPE)", cuda_config)
        self.assertIn('VERBOSE="${VERBOSE:-false}"', build_script)
        self.assertIn("-DSD_VERBOSE_BUILD=$SD_VERBOSE_BUILD", build_script)
        self.assertIn('DL4J_NVCC_FILTER_VERBOSE', nvcc_filter)
        self.assertIn('if verbose_enabled():', nvcc_filter)
        self.assertNotIn('_log_count', nvcc_filter)
        self.assertIn('=== NVCC FAILED', nvcc_filter)


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
        dependency_cache = config["dependencyCache"]
        self.assertEqual(
            "https://dl4jrel26302370c1eeb25.blob.core.windows.net/releases",
            dependency_cache["publicBaseUrl"],
        )
        self.assertEqual(
            {"android-arm64", "android-x86_64"},
            {
                target["compatibility"]["javacppPlatform"]
                for target in dependency_cache["targets"]
            },
        )

    def test_legacy_direct_zluda_variant_is_normalized(self):
        config = prepare_worker.worker_config(
            self.args(
                shard="windows-x86_64-zluda",
                variant="zluda",
            )
        )
        self.assertEqual(
            ["cuda-12.9"],
            [variant["name"] for variant in config["shard"]["build"]["variants"]],
        )

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
