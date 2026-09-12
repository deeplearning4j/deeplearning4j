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
                   EXPECTED_COMMIT="a" * 40, REGENERATE_OPS="false")
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

    def test_release_javadoc_uses_original_sources_without_delombok(self):
        for path in ("pom.xml",
                     "nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/pom.xml",
                     "nd4j/nd4j-ggml/pom.xml", "nd4j/samediff-llm/pom.xml",
                     "nd4j/samediff-vlm/pom.xml"):
            with self.subTest(pom=path):
                pom = ET.parse(ROOT / path).getroot()
                for plugin in pom.findall(".//m:plugin", NS):
                    artifact = plugin.findtext("m:artifactId", namespaces=NS)
                    self.assertNotEqual("lombok-maven-plugin", artifact)
                    if artifact == "maven-javadoc-plugin":
                        for config in plugin.findall(".//m:configuration", NS):
                            self.assertIsNone(config.find("m:sourcepath", NS))
                            self.assertIsNone(config.find("m:doclint", NS))
                            self.assertIsNone(config.find("m:excludePackageNames", NS))
                            self.assertIn(config.findtext("m:failOnError", namespaces=NS), (None, "true"))
                            self.assertIn(config.findtext("m:skip", namespaces=NS), (None, "false"))

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

    def test_java_regeneration_requires_explicit_build_only_mode(self):
        script = step_script("java-hotfix-release.yml", "Validate publication inputs and credentials")
        credentials = {name: f"sensitive-{name}" for name in SECRET_NAMES}
        for mode in ("0", "1"):
            for dry_run in ("true", "false"):
                with self.subTest(mode=mode, dry_run=dry_run):
                    result, _ = self.run_script(script, DEPLOY_TO_RELEASE_STAGING=mode,
                                                DRY_RUN=dry_run, REGENERATE_OPS="true", **credentials)
                    self.assertEqual(0 if dry_run == "true" else 2, result.returncode, result.stderr)
                    if dry_run == "false":
                        self.assertIn("regenerateOps requires dryRun=true", result.stderr)
        for value in ("TRUE", "1", ""):
            with self.subTest(value=value):
                result, _ = self.run_script(script, DRY_RUN="true", REGENERATE_OPS=value)
                self.assertEqual(2, result.returncode)
                self.assertIn("regenerateOps must be true or false", result.stderr)

    def test_java_regeneration_is_remote_reviewable_and_never_a_deployment(self):
        entrypoint = (WORKFLOWS / "build-deploy-cross-platform.yml").read_text()
        java_job = entrypoint.split("  java-only:\n", 1)[1].split("\n  release:\n", 1)[0]
        self.assertIn("if: inputs.javaOnly == '1'", java_job)
        self.assertIn("regenerateOps: ${{ inputs.regenerateOps || 'false' }}", java_job)
        dry_run_input = entrypoint.split("      dryRun:\n", 1)[1].split("      publishSourceRunId:\n", 1)[0]
        self.assertIn('default: "true"', dry_run_input)
        source = (WORKFLOWS / "java-hotfix-release.yml").read_text()
        regeneration = source.split("      - name: Regenerate op APIs for build-only qualification\n", 1)[1]
        regeneration = regeneration.split("\n      - ", 1)[0]
        self.assertIn("inputs.regenerateOps == 'true' && inputs.dryRun == 'true'", regeneration)
        self.assertIn('[[ "${DRY_RUN}" == true && "${REGENERATE_OPS}" == true ]] || exit 2', regeneration)
        self.assertIn("-pl :op-codegen --also-make install", regeneration)
        self.assertIn("org.codehaus.mojo:exec-maven-plugin:3.3.0:java", regeneration)
        self.assertIn("-namespaces all -projects all", regeneration)
        self.assertNotIn(" deploy", regeneration)
        self.assertNotIn(" clean", regeneration)
        self.assertNotIn("-Pcentral-signing", regeneration)
        self.assertIn("op-codegen-source.sha", regeneration)
        self.assertIn("op-codegen.patch", regeneration)
        self.assertIn("git add --intent-to-add -- nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java", regeneration)
        self.assertLess(source.index("name: Retain generated source patch for review"),
                        source.index("name: Set the requested publication version"))
        self.assertIn("-Pcentral-release", step_script("java-hotfix-release.yml", "Build and deploy Java-only modules"))

    def test_java_artifact_verification_fails_when_required_archive_is_absent(self):
        script = step_script("java-hotfix-release.yml", "Verify built artifacts")
        result, _ = self.run_script('find() { return 0; }\n' + script, DRY_RUN="true")
        self.assertEqual(1, result.returncode, result.stderr)
        self.assertIn("Missing or ambiguous built artifact: nd4j/nd4j-shade/jackson (jackson-1.0.0-M3.jar)", result.stderr)

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
        self.assertNotIn("ARG=<--also-make>", result.stdout)
        self.assertIn("ARG=<-Psdx>", result.stdout)
        # samediff-llm depends on tokenizers-native (JavaCPP API jar) at compile
        # scope, so the tokenizers producers must be in the reactor; their Rust
        # and JNI compilation is skipped for this lane.
        self.assertIn("ARG=<-Ptokenizers-native>", result.stdout)
        self.assertIn("ARG=<-Dlibtokenizers.cpu.compile.skip=true>", result.stdout)
        self.assertIn("ARG=<-Djavacpp.compiler.skip=true>", result.stdout)
        for backend in ("cpu", "cuda", "vulkan", "tpu", "hexagon", "zluda", "metal",
                        "native"):
            self.assertNotIn(f"ARG=<-P{backend}>", result.stdout)
        for native_module in ("libnd4j", "libtokenizers", "tokenizers-native",
                              "blas-lapack-generator", "libnd4j-gen", "nd4j-sdx"):
            self.assertNotIn(f"!:{native_module}", result.stdout)
        self.assertNotIn("ARG=<-Psdx-native>", result.stdout)
        root_pom = ET.parse(ROOT / "pom.xml").getroot()
        active = {m.text for m in root_pom.find("m:modules", NS).findall("m:module", NS)}
        self.assertNotIn("libnd4j", active)
        self.assertNotIn("platform-tests", active)
        for module in ("nd4j", "datavec", "deeplearning4j", "python4j", "omnihub", "codegen"):
            self.assertIn(module, active, module)
        profiles = {p.findtext("m:id", namespaces=NS): {m.text for m in p.findall("m:modules/m:module", NS)}
                    for p in root_pom.findall("m:profiles/m:profile", NS)}
        self.assertEqual({"libnd4j"}, profiles["native"])
        self.assertEqual({"nd4j/nd4j-tokenizers/libtokenizers", "nd4j/nd4j-tokenizers/tokenizers-native-preset",
                          "nd4j/nd4j-tokenizers/tokenizers-native"}, profiles["tokenizers-native"])
        self.assertNotIn("codegen-native", profiles)
        codegen_pom = ET.parse(ROOT / "codegen/pom.xml").getroot()
        codegen_modules = {m.text for m in codegen_pom.find("m:modules", NS).findall("m:module", NS)}
        self.assertEqual({"op-codegen"}, codegen_modules)
        codegen_profiles = {p.findtext("m:id", namespaces=NS): {m.text for m in p.findall("m:modules/m:module", NS)}
                            for p in codegen_pom.findall("m:profiles/m:profile", NS)}
        self.assertEqual({"libnd4j-gen", "blas-lapack-generator"}, codegen_profiles["codegen-native"])
        tokenizers = ET.parse(ROOT / "nd4j/nd4j-tokenizers/pom.xml").getroot()
        self.assertIsNone(tokenizers.find("m:modules", NS))

    SENTINEL_MODULES = (
        "nd4j/nd4j-shade/jackson", "nd4j/nd4j-common",
        "nd4j/nd4j-backends/nd4j-api-parent/nd4j-api",
        "nd4j/nd4j-backends/nd4j-api-parent/nd4j-native-api",
        "nd4j/nd4j-serde/nd4j-arrow", "nd4j/nd4j-ggml",
        "nd4j/nd4j-backends/nd4j-backend-impls/nd4j-sdx-preset",
        "nd4j/nd4j-backends/nd4j-backend-impls/nd4j-sdx-model",
        "nd4j/nd4j-backends/nd4j-backend-impls/nd4j-sdx-litertlm",
        "nd4j/samediff-llm", "nd4j/samediff-vlm", "nd4j/samediff-audio",
        "nd4j/samediff-pipeline-ggml", "datavec/datavec-api",
        "deeplearning4j/deeplearning4j-nn", "deeplearning4j/deeplearning4j-core",
        "deeplearning4j/deeplearning4j-modelimport",
        "deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-ui", "omnihub",
        "codegen/op-codegen", "resources",
    )

    def test_java_verification_covers_every_java_subtree(self):
        """The workflow verification must accept a complete snapshot artifact
        set. Provision minimal jars for every sentinel, run the script, then
        remove exactly what this test created."""
        version = "1.0.0-SNAPSHOT"
        created = []
        try:
            for path in self.SENTINEL_MODULES:
                module = Path(path).name
                target = ROOT / path / "target"
                target.mkdir(parents=True, exist_ok=True)
                jar = target / f"{module}-{version}.jar"
                jar.write_bytes(b"placeholder")
                created.append(jar)
            script = step_script("java-hotfix-release.yml", "Verify built artifacts")
            result, _ = self.run_script(script, DRY_RUN="true", DEPLOY_TO_RELEASE_STAGING="0")
            self.assertEqual(0, result.returncode, result.stderr)
            for sentinel in ("nd4j/nd4j-shade/jackson", "datavec/datavec-api",
                             "deeplearning4j/deeplearning4j-nn", "deeplearning4j/deeplearning4j-core",
                             "deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-ui", "omnihub",
                             "nd4j/nd4j-serde/nd4j-arrow",
                             "nd4j/nd4j-backends/nd4j-api-parent/nd4j-native-api",
                             "codegen/op-codegen", "resources"):
                self.assertIn(sentinel, result.stdout, sentinel)
            self.assertNotIn("deeplearning4j/deeplearning4j-ui/deeplearning4j-ui", result.stdout)
        finally:
            for jar in created:
                jar.unlink(missing_ok=True)
                parent = jar.parent
                if parent.name == "target" and not any(parent.iterdir()):
                    parent.rmdir()

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

    def test_toolchain_downloads_use_existing_tls_retry_policy(self):
        bootstrap = (ROOT / "release/github/bootstrap-worker.sh").read_text()
        # curl --retry alone does not retry SSL connection resets (exit 35).
        # Keep the capability probe: older release containers lack this flag.
        self.assertIn('if curl --retry-all-errors --help', bootstrap)
        self.assertIn('CURL_RETRY_ALL=(--retry-all-errors)', bootstrap)
        self.assertIn('CURL_RETRY_ALL=()', bootstrap)
        for function, timeout in (("ensure_protobuf", 300), ("ensure_protoc_21", 300),
                                  ("ensure_android_ndk", 1800)):
            with self.subTest(function=function):
                body = bootstrap.split(f"{function}() {{\n", 1)[1].split("\n}\n", 1)[0]
                self.assertIn('curl --fail --location --retry 5 "${CURL_RETRY_ALL[@]}"', body)
                self.assertIn(f"--connect-timeout 20 --max-time {timeout}", body)
                self.assertNotIn("--insecure", body)

    def test_mlir_bootstrap_is_independent_of_triton_selection(self):
        source = (ROOT / "libnd4j/buildnativeoperations.sh").read_text()
        start = source.index('if [ "$MLIR" == "ON" ]; then\n    print_colored')
        end = source.index("\n# The build tool must match", start)
        recorder = (
            'print_colored() { :; }\n'
            'run_cmake_configure_logged() { printf "configure=%s\\n" "$1"; }\n'
            'run_compiler_dependency_bootstrap() { printf "bootstrap\\n"; }\n'
        )
        for mlir in ("ON", "OFF"):
            for triton in ("ON", "OFF"):
                for cmake_only in ("ON", "OFF"):
                    with self.subTest(mlir=mlir, triton=triton, cmake_only=cmake_only):
                        result, _ = self.run_script(
                            recorder + source[start:end], MLIR=mlir,
                            TRITON=triton, CMAKE_ONLY=cmake_only)
                        self.assertEqual(0, result.returncode, result.stderr)
                        expected = ["configure=OFF"]
                        if mlir == "ON":
                            expected = ["configure=ON"]
                            if cmake_only == "OFF":
                                expected += ["bootstrap", "configure=OFF"]
                        self.assertEqual(expected, result.stdout.splitlines())

    def test_release_workers_do_not_substitute_distro_mlir_packages(self):
        for filename in ("github/bootstrap-worker.sh", "aws/worker.sh",
                         "azure/worker.sh", "gcp/worker.sh"):
            with self.subTest(worker=filename):
                source = (ROOT / "release" / filename).read_text()
                self.assertNotIn("llvm-18-dev", source)
                self.assertNotIn("libmlir-dev", source)
                self.assertNotIn("mlir-tools", source)
        bootstrap = (ROOT / "release/github/bootstrap-worker.sh").read_text()
        sbsa = bootstrap.split("ensure_cuda_sbsa_cross() {\n", 1)[1].split("\n}\n", 1)[0]
        self.assertIn("    zlib1g-dev:arm64\n", sbsa)
        self.assertNotIn("continuing without target zlib", sbsa)

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
