"""Offline regressions for the audited Central publication paths.

Run from platform-tests with PYTHONPATH=..; no Maven build or network upload is
performed. Shell tests execute the actual workflow validation/build blocks,
with Maven replaced by an argument-recording shell function.
"""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock
from urllib.parse import parse_qs, urlparse
import xml.etree.ElementTree as ET

from release.central import repository


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github/workflows"
NS = {"m": "http://maven.apache.org/POM/4.0.0"}
SECRET_NAMES = (
    "CENTRAL_SONATYPE_TOKEN_USERNAME", "CENTRAL_SONATYPE_TOKEN_PASSWORD",
    "GPG_PRIVATE_KEY", "GPG_PASSPHRASE",
)
VALIDATORS = (
    ("_release-worker.yml", "Validate publication configuration"),
    ("publish-release-worker-artifacts.yml", "Validate publication credentials before downloading workers"),
    ("java-hotfix-release.yml", "Validate publication inputs and credentials"),
)
POM_METADATA = """
<name>Release test component</name><description>Release metadata fixture</description>
<url>https://example.invalid/project</url>
<licenses><license><name>Apache-2.0</name><url>https://www.apache.org/licenses/LICENSE-2.0</url></license></licenses>
<developers><developer><name>Project contributors</name></developer></developers>
<scm><connection>scm:git:https://example.invalid/project.git</connection>
<developerConnection>scm:git:ssh://example.invalid/project.git</developerConnection>
<url>https://example.invalid/project</url></scm>
"""


def step_script(filename, name):
    workflow = (WORKFLOWS / filename).read_text(encoding="utf-8")
    step = workflow.split(f"      - name: {name}\n", 1)[1].split("\n      - ", 1)[0]
    return textwrap.dedent(step.split("        run: |\n", 1)[1])


class PublicationWorkflowSafetyTests(unittest.TestCase):
    def run_script(self, script, **overrides):
        env = dict(os.environ)
        env.update({name: "" for name in SECRET_NAMES})
        env.update(RELEASE_VERSION="1.0.0-M3", SNAPSHOT_VERSION="1.0.0-SNAPSHOT",
                   DEPLOY_TO_RELEASE_STAGING="1", DRY_RUN="false", SERVER_ID="central",
                   EXPECTED_COMMIT="a" * 40)
        env.update(overrides)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "github-output"
            env["GITHUB_OUTPUT"] = str(output)
            result = subprocess.run(["bash", "-c", script], env=env, cwd=ROOT,
                                    capture_output=True, text=True, check=False)
            outputs = output.read_text() if output.exists() else ""
        return result, outputs

    def test_pom_uses_verified_namespace_and_stages_by_default(self):
        pom = ET.parse(ROOT / "pom.xml").getroot()
        self.assertEqual("org.eclipse.deeplearning4j", pom.findtext("m:groupId", namespaces=NS))
        self.assertEqual("false", pom.findtext("m:properties/m:central.publishing.autoPublish", namespaces=NS))
        plugin = next(p for p in pom.findall("m:build/m:plugins/m:plugin", NS)
                      if p.findtext("m:artifactId", namespaces=NS) == "central-publishing-maven-plugin")
        self.assertEqual("${central.publishing.autoPublish}",
                         plugin.findtext("m:configuration/m:autoPublish", namespaces=NS))
        self.assertEqual("${central.publishing.publishingRepositoryId}",
                         plugin.findtext("m:configuration/m:publishingServerId", namespaces=NS))

    def test_workflows_use_configured_gpg_secrets_and_never_request_auto_release(self):
        for name in ("_release-worker.yml", "publish-release-worker-artifacts.yml",
                     "publish-central-from-release.yml", "java-hotfix-release.yml",
                     "build-deploy-cross-platform.yml"):
            with self.subTest(workflow=name):
                source = (WORKFLOWS / name).read_text()
                self.assertIn("secrets.GPG_PRIVATE_KEY", source)
                self.assertIn("secrets.GPG_PASSPHRASE", source)
                self.assertNotIn("SONATYPE_GPG_KEY", source)
                self.assertNotIn("PACKAGES_GPG_PASS", source)
                self.assertNotIn("--automatic", source)
                self.assertNotIn("inputs.automatic", source)
                self.assertNotIn("autoPublish=true", source)

    def test_release_profiles_generate_metadata_and_sign_only_when_requested(self):
        pom = ET.parse(ROOT / "pom.xml").getroot()
        profiles = {p.findtext("m:id", namespaces=NS): p for p in pom.findall("m:profiles/m:profile", NS)}
        metadata = profiles["central-release"]
        plugins = {p.findtext("m:artifactId", namespaces=NS): p
                   for p in metadata.findall("m:build/m:plugins/m:plugin", NS)}
        self.assertEqual({"maven-source-plugin", "maven-javadoc-plugin"}, set(plugins))
        for artifact, goal in (("maven-source-plugin", "jar-no-fork"), ("maven-javadoc-plugin", "jar")):
            self.assertEqual(goal, plugins[artifact].findtext("m:executions/m:execution/m:goals/m:goal", namespaces=NS))
            self.assertEqual("package", plugins[artifact].findtext("m:executions/m:execution/m:phase", namespaces=NS))
        self.assertEqual("true", plugins["maven-javadoc-plugin"].findtext("m:configuration/m:failOnError", namespaces=NS))
        signing = profiles["central-signing"].find("m:build/m:plugins/m:plugin", NS)
        self.assertEqual("maven-gpg-plugin", signing.findtext("m:artifactId", namespaces=NS))
        self.assertEqual("MAVEN_GPG_PASSPHRASE", signing.findtext("m:configuration/m:passphraseEnvName", namespaces=NS))
        self.assertEqual("verify", signing.findtext("m:executions/m:execution/m:phase", namespaces=NS))
        self.assertIsNone(metadata.find("m:activation", NS))
        self.assertIsNone(profiles["central-signing"].find("m:activation", NS))

    def test_publication_uses_workflow_tooling_without_changing_source_identity_checks(self):
        for name in ("_release-worker.yml", "publish-central-from-release.yml"):
            with self.subTest(workflow=name):
                source = (WORKFLOWS / name).read_text()
                checkout = source.split("      - name: Checkout publication tooling from the workflow revision\n", 1)[1]
                checkout = checkout.split("\n      - ", 1)[0]
                self.assertIn("ref: ${{ github.workflow_sha }}", checkout)
                self.assertIn("persist-credentials: false", checkout)
        source = (WORKFLOWS / "_release-worker.yml").read_text()
        self.assertIn('"${current_commit}" != "${EXPECTED_SOURCE_COMMIT}"', source)
        self.assertIn('--commit "${EXPECTED_COMMIT}"', (WORKFLOWS / "publish-central-from-release.yml").read_text())

    def test_dry_runs_do_not_require_publication_credentials(self):
        for filename, name in VALIDATORS:
            for mode in ("0", "1"):
                with self.subTest(workflow=filename, mode=mode):
                    result, outputs = self.run_script(step_script(filename, name), DRY_RUN="true",
                                                      DEPLOY_TO_RELEASE_STAGING=mode)
                    self.assertEqual(0, result.returncode, result.stderr)
                    if filename == "_release-worker.yml":
                        self.assertIn("publish=false", outputs)

    def test_release_preflight_checks_each_secret_without_disclosing_values(self):
        credentials = {name: f"sensitive-{name}" for name in SECRET_NAMES}
        for filename, name in (*VALIDATORS,
                               ("publish-central-from-release.yml", "Validate immutable publication identity")):
            for missing in SECRET_NAMES:
                with self.subTest(workflow=filename, missing=missing):
                    result, outputs = self.run_script(step_script(filename, name),
                                                      **{**credentials, missing: ""})
                    self.assertEqual(2, result.returncode)
                    self.assertIn(f"Missing required publication secret: {missing}", result.stderr)
                    self.assertNotIn("sensitive-", result.stdout + result.stderr)
                    self.assertNotIn("publish=true", outputs)
            result, _ = self.run_script(step_script(filename, name), **credentials)
            self.assertEqual(0, result.returncode, result.stderr)

    def test_snapshot_preflight_requires_tokens_but_not_signing_key(self):
        for filename, name in VALIDATORS:
            with self.subTest(workflow=filename):
                result, _ = self.run_script(step_script(filename, name), DEPLOY_TO_RELEASE_STAGING="0",
                                            CENTRAL_SONATYPE_TOKEN_USERNAME="token-user",
                                            CENTRAL_SONATYPE_TOKEN_PASSWORD="token-password")
                self.assertEqual(0, result.returncode, result.stderr)

    def test_invalid_modes_fail_even_for_dry_run(self):
        for filename, name in VALIDATORS:
            for overrides in ({"DRY_RUN": "FALSE"}, {"DEPLOY_TO_RELEASE_STAGING": "2"}):
                with self.subTest(workflow=filename, overrides=overrides):
                    result, _ = self.run_script(step_script(filename, name), **overrides)
                    self.assertEqual(2, result.returncode)

    def test_java_build_uses_literal_arguments_and_never_auto_publishes(self):
        source = (WORKFLOWS / "java-hotfix-release.yml").read_text()
        script = step_script("java-hotfix-release.yml", "Build and deploy Java-only modules")
        self.assertNotIn("eval ", script)
        self.assertNotIn("${{ inputs.", script)
        self.assertLess(source.index("name: Validate publication inputs and credentials"),
                        source.index("name: Set up Java for publishing"))
        recorder = 'mvn() { printf "ARG=<%s>\\n" "$@"; }\n'
        for dry_run, goal in (("true", "install"), ("false", "deploy")):
            with self.subTest(dry_run=dry_run):
                result, _ = self.run_script(recorder + script, DRY_RUN=dry_run, SERVER_ID="custom-central")
                self.assertEqual(0, result.returncode, result.stderr)
                self.assertIn(f"ARG=<{goal}>", result.stdout)
                self.assertIn("ARG=<-Dcentral.publishing.autoPublish=false>", result.stdout)
                self.assertIn("ARG=<-Dcentral.publishing.publishingRepositoryId=custom-central>", result.stdout)
                self.assertIn("ARG=<-Pcentral-release>", result.stdout)
                self.assertNotIn("ARG=<-Prelease>", result.stdout)
                if dry_run == "true":
                    self.assertNotIn("ARG=<deploy>", result.stdout)
                    self.assertNotIn("ARG=<-Pcentral-signing>", result.stdout)
                else:
                    self.assertIn("ARG=<-Pcentral-signing>", result.stdout)
        result, _ = self.run_script(recorder + script, DEPLOY_TO_RELEASE_STAGING="0")
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertNotIn("ARG=<-Pcentral-release>", result.stdout)
        self.assertNotIn("ARG=<-Pcentral-signing>", result.stdout)

    def test_release_dry_runs_check_metadata_and_retain_inspectable_repository(self):
        for filename in ("_release-worker.yml", "publish-release-worker-artifacts.yml"):
            source = (WORKFLOWS / filename).read_text()
            self.assertIn("metadata_args+=(--release-metadata)", source)
            self.assertIn("Retain merged Maven repository for build-only inspection", source)
            self.assertIn("always() && steps.staged.outputs.publish == 'false'", source)
            retention = source.split("      - name: Retain merged Maven repository for build-only inspection\n", 1)[1]
            retention = retention.split("\n      - ", 1)[0]
            self.assertIn("include-hidden-files: true", retention)
            self.assertIn("if-no-files-found: error", retention)

    def test_release_merge_does_not_allow_conflicting_unclassified_components(self):
        for filename in ("_release-worker.yml", "publish-release-worker-artifacts.yml"):
            source = step_script(filename, "Merge and verify staged Maven repository")
            start = source.index("merge_command=(")
            end = source.index("repository_count=0", start)
            recorder = 'python3() { printf "ARG=<%s>\\n" "$@"; }\n'
            script = recorder + source[start:end] + '\n"${merge_command[@]}"\n'
            for version, allow in (("1.0.0-M3", False), ("1.0.0-SNAPSHOT", True)):
                with self.subTest(workflow=filename, version=version):
                    result, _ = self.run_script(script, RELEASE_VERSION=version, SOURCE_COMMIT="a" * 40)
                    self.assertEqual(0, result.returncode, result.stderr)
                    self.assertEqual(allow, "ARG=<--allow-unclassified-duplicates>" in result.stdout)

    def test_upload_defaults_to_user_managed_and_stops_at_validation(self):
        with mock.patch.object(sys, "argv", ["repository.py", "upload", "--bundle", "release.zip"]):
            args = repository.parse_args()
        self.assertFalse(args.automatic)
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "release.zip"
            bundle.write_bytes(b"bundle-for-request-contract")
            with mock.patch.object(repository, "request", side_effect=[
                b"deployment-id", b'{"deploymentState":"VALIDATED"}'
            ]) as request:
                result = repository.upload(bundle, "test-user", "test-password", args.automatic, 60)
        self.assertEqual("deployment-id", result)
        self.assertEqual(2, request.call_count)
        query = parse_qs(urlparse(request.call_args_list[0].args[0]).query)
        self.assertEqual(["USER_MANAGED"], query["publishingType"])
        self.assertIn("/publisher/status?", request.call_args_list[1].args[0])


class ReleaseMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def component(self, artifact="nd4j-api", version="1.0.0-M3", group="org.eclipse.deeplearning4j",
                  packaging="jar", parent="", metadata=POM_METADATA):
        directory = self.root / Path(*group.split(".")) / artifact / version
        directory.mkdir(parents=True, exist_ok=True)
        base = directory / f"{artifact}-{version}"
        Path(str(base) + ".pom").write_text(
            f'<project xmlns="http://maven.apache.org/POM/4.0.0"><modelVersion>4.0.0</modelVersion>'
            f'{parent}<groupId>{group}</groupId><artifactId>{artifact}</artifactId><version>{version}</version>'
            f'<packaging>{packaging}</packaging>{metadata}</project>', encoding="utf-8")
        if packaging != "pom":
            for suffix in (f".{packaging}", "-sources.jar", "-javadoc.jar"):
                Path(str(base) + suffix).write_bytes(b"metadata-fixture")
        return base

    def test_complete_local_metadata_and_pom_only_components_pass(self):
        self.component()
        self.component(artifact="deeplearning4j", packaging="pom")
        repository.verify_release_metadata(self.root)

    def test_required_pom_fields_cannot_be_omitted(self):
        for field in ("name", "description", "url", "licenses", "developers",
                      "scm/connection", "scm/developerConnection", "scm/url"):
            with self.subTest(field=field):
                metadata = ET.fromstring(f"<metadata>{POM_METADATA}</metadata>")
                parent_path, _, child = field.rpartition("/")
                parent = metadata.find(parent_path) if parent_path else metadata
                parent.remove(parent.find(child))
                self.component(metadata="".join(ET.tostring(element, encoding="unicode") for element in metadata))
                with self.assertRaisesRegex(ValueError, f"missing Central POM {field}"):
                    repository.verify_release_metadata(self.root)

    def test_blank_license_and_developer_records_are_rejected(self):
        for field, invalid in (("licenses", "<licenses><license/></licenses>"),
                               ("developers", "<developers><developer><name> </name></developer></developers>")):
            with self.subTest(field=field):
                metadata = ET.fromstring(f"<metadata>{POM_METADATA}</metadata>")
                metadata.remove(metadata.find(field))
                metadata.append(ET.fromstring(invalid))
                self.component(metadata="".join(ET.tostring(element, encoding="unicode") for element in metadata))
                with self.assertRaisesRegex(ValueError, f"missing Central POM {field}"):
                    repository.verify_release_metadata(self.root)

    def test_staged_parent_metadata_is_inherited(self):
        self.component(artifact="deeplearning4j", packaging="pom")
        self.component(metadata="", parent='<parent><groupId>org.eclipse.deeplearning4j</groupId>'
                       '<artifactId>deeplearning4j</artifactId><version>1.0.0-M3</version></parent>')
        repository.verify_release_metadata(self.root)

    def test_parent_cycles_are_rejected(self):
        self.component(artifact="deeplearning4j", packaging="pom", parent='<parent>'
                       '<groupId>org.eclipse.deeplearning4j</groupId><artifactId>deeplearning4j</artifactId>'
                       '<version>1.0.0-M3</version></parent>')
        with self.assertRaisesRegex(ValueError, "cyclic parent POM ancestry"):
            repository.verify_release_metadata(self.root)

    def test_aar_and_war_require_documentation_too(self):
        for packaging in ("aar", "war"):
            with self.subTest(packaging=packaging):
                base = self.component(packaging=packaging)
                repository.verify_release_metadata(self.root)
                Path(str(base) + "-javadoc.jar").unlink()
                with self.assertRaisesRegex(ValueError, "missing Central javadoc attachment"):
                    repository.verify_release_metadata(self.root)

    def test_documentation_only_component_is_rejected(self):
        base = self.component()
        Path(str(base) + ".jar").unlink()
        with self.assertRaisesRegex(ValueError, "no main/classifier archive"):
            repository.verify(self.root, None, None, None)

    def test_missing_documentation_is_not_mistaken_for_publishable_release(self):
        for classifier in ("sources", "javadoc"):
            with self.subTest(classifier=classifier):
                base = self.component()
                Path(str(base) + f"-{classifier}.jar").unlink()
                repository.verify(self.root, None, None, None)
                with self.assertRaisesRegex(ValueError, f"missing Central {classifier} attachment"):
                    repository.verify_release_metadata(self.root)

    def test_classifier_only_slice_is_not_a_complete_release_component(self):
        base = self.component()
        Path(str(base) + ".jar").rename(Path(str(base) + "-linux-x86_64.jar"))
        with self.assertRaisesRegex(ValueError, "missing main artifact"):
            repository.verify_release_metadata(self.root)

    def test_legacy_namespace_is_rejected_for_new_release(self):
        self.component(group="org.nd4j")
        with self.assertRaisesRegex(ValueError, "outside org.eclipse.deeplearning4j"):
            repository.verify_release_metadata(self.root)

    def test_snapshot_release_is_rejected(self):
        self.component(version="1.0.0-SNAPSHOT")
        with self.assertRaisesRegex(ValueError, "SNAPSHOT artifact"):
            repository.verify_release_metadata(self.root)

    def test_mismatched_pom_coordinates_are_rejected(self):
        base = self.component()
        pom = Path(str(base) + ".pom")
        pom.write_text(pom.read_text().replace("<artifactId>nd4j-api", "<artifactId>wrong-artifact"))
        with self.assertRaisesRegex(ValueError, "coordinates do not match"):
            repository.verify_release_metadata(self.root)

    def test_same_release_parent_must_be_present(self):
        self.component(parent='<parent><groupId>org.eclipse.deeplearning4j</groupId>'
                       '<artifactId>deeplearning4j</artifactId><version>1.0.0-M3</version></parent>')
        with self.assertRaisesRegex(ValueError, "missing same-release parent POM"):
            repository.verify_release_metadata(self.root)
        self.component(artifact="deeplearning4j", packaging="pom")
        repository.verify_release_metadata(self.root)

    def test_invalid_metadata_is_rejected_before_signing(self):
        base = self.component()
        Path(str(base) + "-sources.jar").unlink()
        with mock.patch.object(repository.subprocess, "run") as gpg:
            with self.assertRaisesRegex(ValueError, "missing Central sources attachment"):
                repository.sign_bundle(self.root, self.root / "bundle.zip")
        gpg.assert_not_called()
        self.assertFalse((self.root / "bundle.zip").exists())


if __name__ == "__main__":
    unittest.main()
