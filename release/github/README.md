# GitHub release worker

`build-deploy-cross-platform.yml` is the single registered GitHub release dispatcher. Its `workflow` input selects a logical matrix ID from `release/github/workflow-matrix.json`; those IDs retain the historical `build-deploy-*.yml` names but do not require matching physical workflow files. The dispatcher calls `.github/workflows/_release-worker.yml`, which obtains its classifiers from `release/aws/release-plan.json` through the matrix and executes the same `release/aws/build-platform.py` worker used by the cloud release controllers.

Linux jobs run in the container image declared for their canonical shard. Windows and macOS jobs run directly on the matching GitHub-hosted runner. `bootstrap-worker.sh` and `bootstrap-worker.ps1` install host prerequisites that are not supplied by those images.

## Compiler cache

For access to the existing Azure Blob sccache, define the optional repository or organization secret `AZURE_SCCACHE_CONNECTION_STRING`. The connection string is passed to the worker as `SCCACHE_AZURE_CONNECTION_STRING`; it is never written to the generated worker configuration. The cache uses container `releases` and key prefix `deeplearning4j/releases/compiler-cache/v1`.

When that secret is unavailable, such as on an untrusted fork, the existing GitHub Actions sccache setup is used as the fallback.

## Sonatype publication

The one dispatcher retains the historical snapshot-versus-release switch:

- `deployToReleaseStaging=0` builds `snapshotVersion`, merges the selected logical matrix once, and publishes it to `https://central.sonatype.com/repository/maven-snapshots/`.
- `deployToReleaseStaging=1` rewrites the same source snapshot to `releaseVersion`, builds and merges the selected logical matrix, signs the merged repository with `GPG_PRIVATE_KEY`, and uploads it to Central Portal as a user-managed deployment. It stops after validation for manual review; it never requests automatic publication.
- `dryRun=true` builds the **selected logical matrix**, merges and verifies its repository, and does not upload to Sonatype. The merged repository is retained as an Actions artifact for seven days, including hidden directory contents. A logical matrix is not the full release: the dispatcher's default matrix currently contains only four CPU base variants.

The release plumbing reuses `release/aws/build-platform.py` for version setup and `release/central/repository.py` for merge, signing, snapshot deployment, and release staging. A failed publication can be retried without rebuilding by setting `publishSourceRunId` to the original workflow run, retaining the same `deployToReleaseStaging` mode. Publication tooling comes from the workflow revision; worker manifests must still match the immutable build source SHA and version.

Non-SNAPSHOT merges reject conflicting duplicate artifacts, rather than choosing the first POM or binary. Release verification additionally checks the `org.eclipse.deeplearning4j` namespace, POM coordinates, local parent closure, required POM metadata, main artifacts, and sources/javadoc attachments. Signing repeats these checks before invoking GPG. These are local prerequisite checks, **not** proof of full module/classifier coverage, effective Maven dependency resolution, signing validity, or Central namespace authorization.

Actual snapshot upload requires `CENTRAL_SONATYPE_TOKEN_USERNAME` and `CENTRAL_SONATYPE_TOKEN_PASSWORD`. Release staging also requires `GPG_PRIVATE_KEY` and `GPG_PASSPHRASE`. Build-only runs do not require publication credentials. Secret names being configured does not prove the key is valid or the account can publish the namespace.

The root `central-release` profile generates sources and javadoc attachments. The separate `central-signing` profile signs during `verify`; it is used by Java-only release deployment, not by build-only runs. Native repository workers sign the merged repository with `repository.py`. Both publication paths default to manual Central review; no GitHub workflow requests automatic release.

For example, build one logical matrix for repository inspection **without uploading** (this is not a full release):

```bash
gh workflow run build-deploy-cross-platform.yml \
  --ref ag_new_release_updates_2 \
  -f workflow=build-deploy-linux-x86_64.yml \
  -f commitId=FULL_40_CHARACTER_SHA \
  -f deployToReleaseStaging=1 \
  -f releaseVersion=1.0.0-M3 \
  -f snapshotVersion=1.0.0-SNAPSHOT \
  -f dryRun=true
```

## Full-release readiness gate

The current dispatcher is suitable for selected native snapshot matrices, but does not yet assemble a complete release repository. Do not equate a successful selected matrix with a release-ready build. Before staging a full release:

1. Add an explicit full-release selection covering the canonical plan (currently 66 native variants in 26 shards), plus an inventory check that rejects missing workers. There is currently no `all` logical matrix.
2. Give shared Java artifacts and parent POMs one explicit build owner. All current CPU workers disable `buildCrossPlatformJava`; the Java hotfix workflow is a separate partial reactor, not a repository-assembly owner. Generate sources/javadoc for every applicable published component, including native binding components; copying existing metadata is not generation.
3. Resolve the CUDA shared-GAV contract: CUDA 12.6, 12.9, and 13.1 rewrite different CUDA dependencies into the same `nd4j-cuda-backend-common` POM coordinate. A full release must not arbitrarily select one. Define a compatible common dependency contract or distinct coordinates before merging.
4. Reconcile platform-module POM dependencies with the actual classifier inventory. In particular, `nd4j-native-platform` still references targets absent from the current plan. Validate same-release parents and dependencies against the assembled repository, not cached snapshots.
5. Complete the whole matrix at one source SHA and verify its attested Maven and SDK outputs. Measure the actual signed ZIP size against Central's 1 GB bundle limit and design component-preserving partitioning if necessary; raw shard sizes alone are not a compressed bundle measurement.
6. Verify signing and namespace authorization, then upload only a user-managed deployment. Require Central `VALIDATED` and human review before publication. Consider protected-environment approval in addition to the manual Portal release step.

The safe first full run should be `deployToReleaseStaging=1` with `dryRun=true` after these assembly gaps are closed. No release should be uploaded merely to discover missing components.

## Matrix maintenance

Add or change classifiers in the provider release plans first. Keep `release/github/workflow-matrix.json` limited to workflow-to-shard and shard-to-runtime mappings. `release/github/test_worker.py` rejects workflow rows that do not resolve to an explicit release-plan variant and checks the historically distinct Linux compile ISA classifiers.

A normal dispatch sets `workflow` to the desired logical matrix ID and runs that complete canonical matrix. Partial reruns must set `targetedRetry=1` and provide one or more exact published classifier IDs in `classifiers`; filters are rejected otherwise. This keeps recovery of a failed classifier explicit without allowing an intended complete release run to silently omit base or another variant.

ZLUDA is selected by its published CUDA/ROCm classifier, such as `linux-x86_64-cuda-12.9-zluda-rocm-7.2.4`. Published worker IDs and Maven classifiers use single hyphens and are the only supported classifier interface.

These workflows run the shared worker locally on GitHub runners; they do not provision AWS, Azure, or GCP virtual machines.
