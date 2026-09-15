# Shared OPEN compatibility staging

These scoped release workflows use `ossrh.py`, not Central Portal bundle uploads.
Snapshots and Cloudflare R2 worker/cache routing are unchanged. An OPEN repository
is not a Portal deployment: it is not visible in Portal deployments until closed.
Nothing here closes, releases, promotes, drops, or manually transfers it to Portal.

## Actions operator sequence

The cross-platform dispatcher already uses 24 of GitHub's 25 permitted inputs.
Its new `staging` string is JSON with `action`, `stagingProfileId`, and
`stagingRepositoryId` fields; existing snapshot and recovery inputs stay intact.
The reusable/direct Java and recovery workflows expose the two IDs as ordinary
named string inputs.

1. Dispatch `build-deploy-cross-platform.yml` with
   `staging={"action":"profiles"}`.
   This only lists the profiles accessible to the existing
   `CENTRAL_SONATYPE_TOKEN_USERNAME` / `CENTRAL_SONATYPE_TOKEN_PASSWORD` secrets.
   Read the service-returned ID. Do not assume an old OSSRH hex ID or a namespace
   is the correct profile; the authenticated service is authoritative.
2. Dispatch the same workflow with `dryRun=false` and
   `staging={"action":"open","stagingProfileId":"<returned profile ID>"}`.
   Leave `stagingRepositoryId` absent/empty. No builds run.
   The returned repository ID is persisted in the `open-staging-<run ID>` artifact
   (`open-staging.json`, retained 90 days), logs, and job summary. Copy the receipt
   to the release's durable operator record before artifact expiry.
3. For ALL subsequent cross-platform runs set
   `staging={"action":"none","stagingProfileId":"<same profile>","stagingRepositoryId":"<same repository>"}`,
   `deployToReleaseStaging=1`, `dryRun=false`.
   Java: `javaOnly=1`. Native CPU: use the existing canonical matrix/classifier
   inputs for Windows/Linux/macOS, or `publishSourceRunId` + artifact selection
   for already-built workers. Both reusable worker publishers and the directly
   dispatchable recovery and Java workflows accept these IDs explicitly.
   Existing releaseVersion/snapshotVersion/source identity inputs still apply.
4. The jobs check authenticated profile membership and OPEN/non-transitioning
   status before building/downloading and again before/after uploading. They
   never silently open a replacement repository. Missing, wrong, closed, or
   inaccessible IDs fail the run. All lanes must use the SAME Central user;
   compatibility repositories are per-user, not merely per-namespace.
5. Inspect the upload log and receipt. Repeated dispatches with different
   classifiers append to this same ID. Avoid overlapping duplicate coordinates;
   Maven image upload is not transactional and an interrupted upload may be
   partial. Upload jobs share a concurrency group keyed by repository ID. GitHub
   concurrency allows only one pending job and may replace pending runs; dispatch
   serially and verify all intended runs actually completed.

Opening is a non-idempotent POST. Do NOT blindly retry an uncertain open request:
inspect authenticated service state first. The ID is persisted immediately upon
receiving it, before the follow-up status check. There is no automatic ID selected
from the current IP or latest repository. No credentials go into the receipt.

## Artifact handling

Java uses `install` plus the explicit `maven-deploy-plugin:3.1.4:deploy` **goal** to
an alternate filesystem repository, never the release `deploy` lifecycle that
would invoke Central's injected publishing extension. `maven.deploy.skip=false`
ensures all reactor parent POMs are included. Before upload the install-log POM
inventory (including known parent sentinels) must match the image byte-for-byte.
Snapshots retain their existing `deploy` lifecycle and `skipPublishing=false`.

Both Java and native lanes invoke the same prebuilt-image uploader. It validates
release metadata and parent closure, signs without creating a Portal ZIP, then
runs the pinned Nexus `deploy-staged-repository` goal outside the source reactor
with explicit profile/repository IDs, `skipStagingRepositoryClose=true`,
`autoReleaseAfterClose=false`, and `keepStagingRepositoryOnFailure=true`.
No build lifecycle or arbitrary Maven flags are accepted by this uploader.

The bootstrap uses Nexus `profiles/<id>/start`, the operation behind `rc-open`.
Status uses `profile_repositories/<id>` and matches the exact `repositoryId`, not
an IP-selected default and not the potentially different manual API repository
key. Errors and unknown response shapes fail closed. If compatibility API
behavior changes, investigate it; do not substitute Portal or implicit uploads.

## Documentation basis (not live publication verification)

- https://central.sonatype.org/publish/publish-portal-ossrh-staging-api/
  explicitly lists Nexus Maven/Gradle plugin compatibility, Central token auth,
  per-user staging, and close as the transfer-to-Portal boundary.
- https://github.com/sonatype/nexus-maven-plugins/blob/main/staging/maven-plugin/README.md
  documents `rc-open`, explicit-ID image uploads and no repository management
  when both IDs are supplied.
- https://github.com/gradle-nexus/publish-plugin/blob/master/src/main/kotlin/io/github/gradlenexus/publishplugin/internal/NexusClient.kt
  provides the supported plugin's JSON start/profile-repository response shapes
  and Basic token authentication used here.

Only documentation was fetched during implementation. No authenticated service
call, staging upload, native build, release, or remote Actions dispatch was run.
Offline tests from `platform-tests`:

`PYTHONPATH=.. python3 -m unittest test_release_publication_safety test_java_snapshot_publication test_open_staging`
