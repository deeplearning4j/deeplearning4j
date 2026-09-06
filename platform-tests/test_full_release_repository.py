"""Full release assembly contracts; run from platform-tests, without native builds."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("full_repository", ROOT / "release/github/full-repository.py")
full = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(full)
VERSION = "1.0.0-M3"
COMMIT = "a" * 40
RUN = "github-123-2"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def pom(artifact, packaging="jar", parent=None):
    parent_xml = (f"<parent><groupId>{full.GROUP}</groupId><artifactId>{parent}</artifactId>"
                  f"<version>{VERSION}</version></parent>") if parent else ""
    return (f'<project xmlns="{full.NS["m"]}"><modelVersion>4.0.0</modelVersion>{parent_xml}'
            f"<groupId>{full.GROUP}</groupId><artifactId>{artifact}</artifactId>"
            f"<version>{VERSION}</version><packaging>{packaging}</packaging></project>")


class FullRepositoryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.plan = full.load_json(ROOT / "release/aws/release-plan.json")
        self.matrix = full.load_json(ROOT / "release/github/workflow-matrix.json")
        self.shards = full.worker.plan_shards(self.plan)

    def receipts(self):
        for shard in self.shards.values():
            for variant in shard["build"]["variants"]:
                selected = copy.deepcopy(shard)
                selected["contractDigest"] = full.contract_digest(shard)
                selected["build"]["variants"] = [copy.deepcopy(variant)]
                selected["build"]["releaseMetadata"] = True
                directory = self.root / "workers" / shard["id"] / variant["name"]
                write_json(directory / "worker-config.json", {
                    "shard": selected, "releaseVersion": VERSION, "commit": COMMIT,
                    "runId": "github-123-1",
                })
                write_json(directory / "build-result.json", {"completedVariants": [variant["name"]]})
                (directory / "worker-success").write_text("success\n")
                (directory / "maven-repository").mkdir()
        return self.root / "workers"

    def test_all_selects_each_canonical_variant_exactly_once(self):
        rows = []
        for group in ("linux", "host"):
            rows.extend(full.worker.workflow_rows(self.plan, self.matrix, "all", group))
        expected = {(shard["id"], variant["name"]) for shard in self.shards.values()
                    for variant in shard["build"]["variants"]}
        self.assertEqual(66, len(expected))
        self.assertEqual(len(expected), len(rows))
        self.assertEqual(expected, {(row["shard"], row["variant"]) for row in rows})
        self.assertEqual(len(rows), len({row["artifactId"] for row in rows}))

    def test_all_rejects_partial_selection(self):
        with self.assertRaises(ValueError):
            full.worker.workflow_rows(self.plan, self.matrix, "all", "linux", classifiers="base")
        with self.assertRaises(ValueError):
            full.worker.workflow_rows(self.plan, self.matrix, "all", "linux",
                                      selection_mode="targeted")

    def test_receipts_require_complete_same_source_same_run_matrix(self):
        workers = self.receipts()
        found = full.inspect_workers(workers, self.plan, VERSION, COMMIT, RUN)
        self.assertEqual(66, len(found))
        receipt = next(workers.rglob("worker-config.json"))
        original = full.load_json(receipt)
        for field, value in (("commit", "b" * 40), ("releaseVersion", "1.0.0-SNAPSHOT"),
                             ("runId", "github-999-1"), ("runId", "github-123-3")):
            with self.subTest(field=field, value=value):
                changed = copy.deepcopy(original)
                changed[field] = value
                write_json(receipt, changed)
                with self.assertRaises(ValueError):
                    full.inspect_workers(workers, self.plan, VERSION, COMMIT, RUN)
        write_json(receipt, original)
        (receipt.parent / "worker-success").unlink()
        with self.assertRaisesRegex(ValueError, "success receipt"):
            full.inspect_workers(workers, self.plan, VERSION, COMMIT, RUN)

    def test_contract_digest_variant_metadata_and_completion_are_checked(self):
        workers = self.receipts()
        receipt = next(workers.rglob("worker-config.json"))
        original = full.load_json(receipt)
        for mutation in ("digest", "variant", "metadata"):
            changed = copy.deepcopy(original)
            if mutation == "digest":
                changed["shard"]["contractDigest"] = "bad"
            elif mutation == "variant":
                changed["shard"]["build"]["variants"][0]["unexpected"] = True
            else:
                changed["shard"]["build"]["releaseMetadata"] = False
            write_json(receipt, changed)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                full.inspect_workers(workers, self.plan, VERSION, COMMIT, RUN)
        write_json(receipt, original)
        write_json(receipt.parent / "build-result.json", {"completedVariants": []})
        with self.assertRaisesRegex(ValueError, "attest completion"):
            full.inspect_workers(workers, self.plan, VERSION, COMMIT, RUN)

    def test_missing_and_duplicate_workers_are_rejected(self):
        workers = self.receipts()
        receipt = next(workers.rglob("worker-config.json"))
        config = full.load_json(receipt)
        receipt.unlink()
        with self.assertRaisesRegex(ValueError, "missing canonical workers"):
            full.inspect_workers(workers, self.plan, VERSION, COMMIT, RUN)
        write_json(receipt, config)
        write_json(workers / "zz-duplicate" / "worker-config.json", config)
        with self.assertRaisesRegex(ValueError, "duplicate worker"):
            full.inspect_workers(workers, self.plan, VERSION, COMMIT, RUN)

    def test_retry_attempt_identity_is_bounded(self):
        self.assertTrue(full.same_workflow_run("github-123-1", RUN))
        self.assertTrue(full.same_workflow_run(RUN, RUN))
        for value in ("github-124-1", "github-123-3", "github-123-0", "", "123"):
            self.assertFalse(full.same_workflow_run(value, RUN))

    def test_seed_native_has_unique_owners_and_only_actual_parents(self):
        workers = self.receipts()
        found = full.inspect_workers(workers, self.plan, VERSION, COMMIT, RUN)
        owners = full.shared_owners(self.plan)
        expected = set()
        for key, repository in found.items():
            shard = self.shards[key[0]]
            variant = next(v for v in shard["build"]["variants"] if v["name"] == key[1])
            build, rules = full.selected_contract(shard, variant)
            for artifact in full.classifier_components(build, rules):
                classifier = full.driver.variant_artifact_classifier_for(build, variant, artifact)
                relative = full.component_path(artifact, VERSION) / f"{artifact}-{VERSION}-{classifier}.jar"
                self.assertNotIn(relative, expected, f"multiple classifier owners: {key}")
                expected.add(relative)
                (repository / relative).parent.mkdir(parents=True, exist_ok=True)
                (repository / relative).write_bytes(b"native fixture")
            for artifact, owner in owners.items():
                if owner != key:
                    continue
                directory = repository / full.component_path(artifact, VERSION)
                directory.mkdir(parents=True, exist_ok=True)
                for suffix in (".pom", ".jar", "-sources.jar", "-javadoc.jar"):
                    path = directory / f"{artifact}-{VERSION}{suffix}"
                    path.write_text(pom(artifact, parent="release-parent") if suffix == ".pom" else "fixture")
        for artifact in ("release-parent", "libnd4j"):
            path = found[full.CPU_OWNER] / full.component_path(artifact, VERSION) / f"{artifact}-{VERSION}.pom"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(pom(artifact, "pom"))
        with patch.object(full.driver, "attest_variant_classifier_artifacts") as variants, \
                patch.object(full.driver, "attest_classifier_archive_contract") as archives:
            local = self.root / "m2"
            inventory = full.seed_native(found, self.plan, local, VERSION)
        self.assertEqual(66, variants.call_count)
        self.assertEqual(len(expected), archives.call_count)
        self.assertTrue(expected.issubset({Path(p) for p in inventory}))
        self.assertFalse(any("/libnd4j/" in p for p in inventory))
        for cuda in ("12.6", "12.9", "13.1"):
            self.assertIn(f"nd4j-cuda-{cuda}-backend-common", owners)
        self.assertTrue(all((local / path).is_file() for path in inventory))

    def test_native_collision_is_rejected_before_java_install(self):
        local = self.root / "m2"
        local.mkdir()
        relative = full.component_path("nd4j-native", VERSION) / f"nd4j-native-{VERSION}.pom"
        model = self.root / "effective.xml"
        model.write_text(pom("nd4j-native"))
        with patch.object(full, "run") as run:
            with self.assertRaisesRegex(ValueError, "native-owned"):
                full.install_java(["mvn", "install"], self.root, {}, model, local, VERSION,
                                  {relative.as_posix(): "native"})
        self.assertEqual(1, run.call_count)
        self.assertNotIn("install", run.call_args.args[0])

    def test_java_install_requires_fresh_main_and_documentation(self):
        local = self.root / "m2"
        directory = local / full.component_path("java-module", VERSION)
        directory.mkdir(parents=True)
        model = self.root / "effective.xml"
        model.write_text(pom("java-module"))
        inventory = {}
        with patch.object(full, "run"):
            for suffix in (".pom", ".jar", "-sources.jar", "-javadoc.jar"):
                with self.assertRaisesRegex(ValueError, "missing required output"):
                    full.install_java(["mvn", "install"], self.root, {}, model, local, VERSION, inventory)
                (directory / f"java-module-{VERSION}{suffix}").write_text("fixture")
            full.install_java(["mvn", "install"], self.root, {}, model, local, VERSION, inventory)
        self.assertEqual({full.JAVA_OWNER}, set(inventory.values()))
        self.assertEqual(4, len(inventory))

    def test_cuda_platform_versions_come_from_immutable_plan(self):
        with patch.object(full, "run") as run, patch.object(full, "install_java") as install, \
                patch.object(full.subprocess, "check_output", return_value="mvn install"):
            full.build_java(self.root, self.root / "m2", self.root, VERSION,
                            "1.0.0-SNAPSHOT", {}, self.plan)
        targets = [call.args[0][-1] for call in run.call_args_list
                   if call.args[0][:2] == ["bash", "./change-cuda-versions.sh"]]
        self.assertEqual(["12.9", "12.6", "12.9", "13.1"], targets)
        self.assertEqual(7, install.call_count)
        for call in install.call_args_list[1:]:
            self.assertNotIn("--also-make", call.args[0])
            self.assertFalse(any(arg.startswith("-Djavacpp.platform=") for arg in call.args[0]))

    def test_consumers_resolve_independently_and_reject_remote_dl4j(self):
        local = self.root / "m2"
        inventory = {}
        for artifact in ("a", "b"):
            relative = full.component_path(artifact, VERSION) / f"{artifact}-{VERSION}.jar"
            (local / relative).parent.mkdir(parents=True, exist_ok=True)
            (local / relative).write_bytes(b"fixture")
            inventory[relative.as_posix()] = full.JAVA_OWNER
        with patch.object(full, "run") as run:
            full.verify_consumer(local, self.root, inventory, self.root)
        models = list((self.root / "consumers").glob("consumer-*/pom.xml"))
        self.assertEqual(2, len(models))
        for model in models:
            self.assertEqual(1, len(ET.parse(model).getroot().findall("m:dependencies/m:dependency", full.NS)))
        self.assertIn(f"-Dmaven.repo.local={local}", run.call_args.args[0])
        extra = local / full.component_path("not-built", VERSION) / f"not-built-{VERSION}.pom"
        extra.parent.mkdir(parents=True)
        extra.write_text(pom("not-built"))
        second = self.root / "second"
        second.mkdir()
        with patch.object(full, "run"), self.assertRaisesRegex(ValueError, "not built by this run"):
            full.verify_consumer(local, second, inventory, self.root)

    def test_contracts_gate_native_fanout(self):
        workflow = (ROOT / ".github/workflows/_release-worker.yml").read_text()
        matrix_job = workflow.split("\n  linux:", 1)[0]
        self.assertIn("name: Validate full repository contracts before native fanout", matrix_job)
        self.assertIn("working-directory: platform-tests", matrix_job)
        self.assertIn("test_full_release_repository test_release_publication_safety", matrix_job)
        self.assertIn("set -Eeuo pipefail", matrix_job)
        self.assertNotIn("continue-on-error", matrix_job)
        self.assertIn("needs: matrix", workflow.split("\n  linux:", 1)[1].split("steps:", 1)[0])
        self.assertIn("needs: matrix", workflow.split("\n  host:", 1)[1].split("steps:", 1)[0])

    def test_workflow_full_assembly_is_distinct_from_partial_merge(self):
        workflow = (ROOT / ".github/workflows/_release-worker.yml").read_text()
        self.assertIn("name: Assemble and verify full Maven repository\n        if: inputs.workflow == 'all'", workflow)
        self.assertIn("name: Merge and verify staged Maven repository\n        if: inputs.workflow != 'all'", workflow)
        self.assertIn('--run-id "github-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"', workflow)
        self.assertIn("path: .release-source", workflow)
        self.assertIn("ref: ${{ needs.matrix.outputs.source }}\n          path: .release-source", workflow)
        self.assertNotIn("--automatic", workflow)


if __name__ == "__main__":
    unittest.main()
