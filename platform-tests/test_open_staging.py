"""Offline OPEN compatibility staging contracts; no Maven/network execution."""
import json
import re
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from release.central import ossrh
import test_release_publication_safety as safety
from test_release_publication_safety import step_script, WORKFLOWS


PROFILE = "org.eclipse.deeplearning4j"
REPO = "service-issued-repo-123"
OPEN = {"repositoryId": REPO, "type": "open", "transitioning": False}


class OpenStagingTests(unittest.TestCase):
    def test_exact_profile_and_repository_are_required(self):
        for profile, repo in (("", REPO), (PROFILE, ""), (PROFILE, "a,b"),
                              ("../profile", REPO), (PROFILE, "repo\n")):
            with self.subTest(profile=profile, repository=repo), mock.patch.object(ossrh, "request") as request:
                with self.assertRaises(ValueError):
                    ossrh.check_open(profile, repo)
                request.assert_not_called()

    def test_profile_scoped_exact_open_lookup_no_ip_selection(self):
        with mock.patch.object(ossrh, "request", return_value={"data": [OPEN]}) as request:
            self.assertEqual(OPEN, ossrh.check_open(PROFILE, REPO))
        request.assert_called_once_with("profile_repositories/" + PROFILE)

    def test_wrong_closed_released_transitioning_or_ambiguous_repository_fails(self):
        for rows in ([], [{**OPEN, "repositoryId": "other"}], [OPEN, OPEN],
                     [{**OPEN, "type": "closed"}], [{**OPEN, "type": "released"}],
                     [{**OPEN, "transitioning": True}],
                     [{"repositoryId": REPO, "type": "open"}]):
            with self.subTest(rows=rows), mock.patch.object(ossrh, "request", return_value={"data": rows}):
                with self.assertRaises(ValueError):
                    ossrh.check_open(PROFILE, REPO)

    def test_open_uses_returned_id_and_persists_before_postcheck(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "identity.json"
            with mock.patch.object(ossrh, "request", side_effect=[
                {"data": [{"id": PROFILE}]}, {"data": {"stagedRepositoryId": REPO}},
                {"data": [OPEN]},
            ]) as request:
                self.assertEqual(REPO, ossrh.open_repository(PROFILE, path))
            self.assertEqual(REPO, json.loads(path.read_text())["stagingRepositoryId"])
            self.assertEqual([
                mock.call("profiles"),
                mock.call("profiles/" + PROFILE + "/start", {"data": {"description": "DL4J shared open staging"}}),
                mock.call("profile_repositories/" + PROFILE),
            ], request.call_args_list)
            with mock.patch.object(ossrh, "request", side_effect=[
                {"data": [{"id": PROFILE}]}, {"data": {"stagedRepositoryId": REPO}},
                OSError("status request failed"),
            ]):
                with self.assertRaises(OSError):
                    ossrh.open_repository(PROFILE, path)
            self.assertEqual(REPO, json.loads(path.read_text())["stagingRepositoryId"])

    def test_open_rejects_unknown_profile_without_post(self):
        with mock.patch.object(ossrh, "request", return_value={"data": []}) as request:
            with self.assertRaises(ValueError):
                ossrh.open_repository(PROFILE, Path("unused.json"))
        self.assertEqual([mock.call("profiles")], request.call_args_list)

    def test_upload_only_invokes_pinned_image_goal_with_no_close_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "image"
            repository.mkdir()
            output = Path(directory) / "receipt.json"
            with mock.patch.object(ossrh, "check_open") as check, \
                 mock.patch.object(ossrh, "sign_repository") as sign, \
                 mock.patch.object(ossrh, "resolve_maven_command", return_value=["mvn"]), \
                 mock.patch.object(ossrh.subprocess, "run") as run:
                ossrh.upload(repository, PROFILE, REPO, output)
            sign.assert_called_once_with(repository)
            self.assertEqual([mock.call(PROFILE, REPO)] * 3, check.call_args_list)
            command = run.call_args.args[0]
            self.assertIn(ossrh.PLUGIN + ":deploy-staged-repository", command)
            for flag in ("-DstagingProfileId=" + PROFILE, "-DstagingRepositoryId=" + REPO,
                         "-DskipStagingRepositoryClose=true", "-DautoReleaseAfterClose=false",
                         "-DkeepStagingRepositoryOnFailure=true"):
                self.assertIn(flag, command)
            self.assertNotIn("deploy", command)
            self.assertNotIn("rc-open", " ".join(command))
            self.assertNotIn("rc-close", " ".join(command))
            self.assertNotEqual(repository, Path(run.call_args.kwargs["cwd"]))
            self.assertEqual(REPO, json.loads(output.read_text())["stagingRepositoryId"])

    def test_resigning_refreshes_signature_checksums_without_a_bundle(self):
        from release.central import repository as image
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "component.pom"
            path.write_text("<project/>")
            signature = Path(str(path) + ".asc")
            signature.write_text("old signature")
            image.write_checksums([signature])
            def sign(command, **kwargs):
                signature.write_text("new signature")
            with mock.patch.object(image, "verify_release_metadata"), \
                 mock.patch.object(image, "primary_files", return_value=[path]), \
                 mock.patch.object(image.subprocess, "run", side_effect=sign):
                image.sign_repository(Path(directory))
            for algorithm in image.CHECKSUMS:
                self.assertEqual(image.digest(signature, algorithm),
                                 Path(str(signature) + "." + algorithm).read_text().strip())
            self.assertFalse(list(Path(directory).glob("*.zip")))

    def test_upload_failure_never_attempts_lifecycle_cleanup(self):
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(ossrh, "check_open"), \
             mock.patch.object(ossrh, "sign_repository"), \
             mock.patch.object(ossrh, "resolve_maven_command", return_value=["mvn"]), \
             mock.patch.object(ossrh.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "mvn")) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                ossrh.upload(Path(directory), PROFILE, REPO, Path(directory) / "receipt.json")
            run.assert_called_once()
            self.assertFalse((Path(directory) / "receipt.json").exists())

    def test_invalid_target_blocks_signing_and_maven(self):
        with mock.patch.object(ossrh, "request", return_value={"data": []}), \
             mock.patch.object(ossrh, "sign_repository") as sign, \
             mock.patch.object(ossrh.subprocess, "run") as run:
            with self.assertRaises(ValueError):
                ossrh.upload(Path("unused"), PROFILE, REPO, Path("unused.json"))
            sign.assert_not_called()
            run.assert_not_called()

    def test_all_installed_java_parent_poms_must_be_staged_byte_identically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lines = []
            for artifact in ("nd4j-backends", "nd4j-api-parent", "nd4j-backend-impls", "extra-parent"):
                relative = f"org/eclipse/deeplearning4j/{artifact}/1.0/{artifact}-1.0.pom"
                local = root / "repository" / relative
                staged = root / "staged" / relative
                for path in (local, staged):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("<project/>")
                lines.append(f"[INFO] Installing pom.xml to {local}")
            log = root / "install.log"
            log.write_text("\n".join(lines))
            ossrh.verify_java_parents(root / "staged", log, "1.0")
            staged.unlink()
            with self.assertRaisesRegex(ValueError, "extra-parent"):
                ossrh.verify_java_parents(root / "staged", log, "1.0")

    def test_workflow_lanes_forward_same_explicit_identity(self):
        caller = (WORKFLOWS / "build-deploy-cross-platform.yml").read_text()
        for job in ("java-only", "release", "publish-staged"):
            section = re.split(r"\n  [a-z][a-z-]*:\n", caller.split(f"  {job}:\n", 1)[1])[0]
            for field in ("stagingProfileId", "stagingRepositoryId"):
                self.assertIn("      " + field + ": ${{ fromJSON(inputs.staging || '{}')." + field + " }}", section)
        for name in ("java-hotfix-release.yml", "_release-worker.yml", "publish-release-worker-artifacts.yml"):
            source = (WORKFLOWS / name).read_text()
            self.assertIn("ossrh.py upload", source)
            self.assertIn("STAGING_PROFILE_ID: ${{ inputs.stagingProfileId }}", source)
            self.assertIn("STAGING_REPOSITORY_ID: ${{ inputs.stagingRepositoryId }}", source)
            self.assertIn("ossrh.py check", source)
            self.assertIn("group: ossrh-upload-${{ inputs.stagingRepositoryId || github.run_id }}", source)
            self.assertNotIn("repository.py upload", source)
            self.assertNotIn("sign-bundle", source)
            self.assertNotIn("--bundle", source)

    def test_java_release_uses_local_explicit_goal_snapshot_stays_deploy(self):
        script = step_script("java-hotfix-release.yml", "Build and deploy Java-only modules")
        recorder = 'mvn() { printf "ARG=<%s>\\n" "$@"; }\n'
        runner = safety.PublicationWorkflowSafetyTests()
        with tempfile.TemporaryDirectory() as directory:
            for mode in ("0", "1"):
                result, _ = runner.run_script(recorder + script, cwd=directory,
                                               DEPLOY_TO_RELEASE_STAGING=mode)
                self.assertEqual(0, result.returncode, result.stderr)
                if mode == "1":
                    self.assertNotIn("ARG=<deploy>", result.stdout)
                    self.assertIn("maven-deploy-plugin:3.1.4:deploy", result.stdout)
                    self.assertIn("-DaltDeploymentRepository=local::file://", result.stdout)
                    self.assertIn("-Dmaven.deploy.skip=false", result.stdout)
                else:
                    self.assertIn("ARG=<deploy>", result.stdout)
                    self.assertNotIn("altDeploymentRepository", result.stdout)
                    self.assertIn("-DskipPublishing=false", result.stdout)

    def test_java_and_cpu_publisher_scripts_use_identical_target_environment(self):
        recorder = ('python3() { printf "TARGET=<%s> PROFILE=<%s>\\n" '
                    '"$STAGING_REPOSITORY_ID" "$STAGING_PROFILE_ID"; '
                    'printf "ARG=<%s>\\n" "$@"; }\n')
        runner = safety.PublicationWorkflowSafetyTests()
        with tempfile.TemporaryDirectory() as directory:
            for folder in (".release-worker-merged", ".release-worker-recovery-merged"):
                (Path(directory) / folder).mkdir()
            for filename, step in (
                ("java-hotfix-release.yml", "Upload Java image to the shared OPEN repository"),
                ("_release-worker.yml", "Upload merged Maven release to shared OPEN staging"),
                ("publish-release-worker-artifacts.yml", "Upload merged Maven release to shared OPEN staging"),
            ):
                result, _ = runner.run_script(recorder + step_script(filename, step), cwd=directory,
                                               GITHUB_WORKSPACE=directory,
                                               STAGING_REPOSITORY_ID=REPO, STAGING_PROFILE_ID=PROFILE)
                self.assertEqual(0, result.returncode, result.stderr)
                self.assertIn(f"TARGET=<{REPO}> PROFILE=<{PROFILE}>", result.stdout)
                self.assertIn("ARG=<release/central/ossrh.py>", result.stdout)
                self.assertIn("ARG=<upload>", result.stdout)
                self.assertNotIn("ARG=<open>", result.stdout)

    def test_compatibility_http_uses_token_auth_and_only_start_post(self):
        import base64
        import io
        with mock.patch.dict(ossrh.os.environ, {
            "CENTRAL_SONATYPE_TOKEN_USERNAME": "user",
            "CENTRAL_SONATYPE_TOKEN_PASSWORD": "secret",
        }), mock.patch.object(ossrh, "build_opener") as opener:
            opener.return_value.open.return_value = io.BytesIO(b'{"data": []}')
            ossrh.request("profiles")
            request = opener.return_value.open.call_args.args[0]
            self.assertEqual("GET", request.get_method())
            self.assertEqual(ossrh.ENDPOINT + "/service/local/staging/profiles", request.full_url)
            self.assertEqual("Basic " + base64.b64encode(b"user:secret").decode(),
                             request.get_header("Authorization"))
            opener.return_value.open.return_value = io.BytesIO(b'{"data": {}}')
            ossrh.request("profiles/" + PROFILE + "/start", {"data": {"description": "test"}})
            request = opener.return_value.open.call_args.args[0]
            self.assertEqual("POST", request.get_method())
            self.assertTrue(request.full_url.endswith("/start"))

    def test_bootstrap_is_explicit_separate_and_receipted(self):
        source = (WORKFLOWS / "build-deploy-cross-platform.yml").read_text()
        self.assertIn('default: \'{"action":"none"}\'', source)
        inputs = source.split("    inputs:\n", 1)[1].split("\npermissions:", 1)[0]
        self.assertEqual(25, len(re.findall(r"^      [A-Za-z][A-Za-z0-9]*:", inputs, re.M)))
        self.assertIn("Opening requires explicit dryRun=false", source)
        self.assertIn("name: open-staging-${{ github.run_id }}", source)
        self.assertEqual(4, source.count("fromJSON(inputs.staging || '{}').action == '' || fromJSON(inputs.staging || '{}').action == 'none'"))


if __name__ == "__main__":
    unittest.main()
