# GitHub release worker

`build-deploy-cross-platform.yml` is the single registered GitHub release dispatcher. Its `workflow` input selects a logical matrix ID from `release/github/workflow-matrix.json`; those IDs retain the historical `build-deploy-*.yml` names but do not require matching physical workflow files. The dispatcher calls `.github/workflows/_release-worker.yml`, which obtains its classifiers from `release/aws/release-plan.json` through the matrix and executes the same `release/aws/build-platform.py` worker used by the cloud release controllers.

Linux jobs run in the container image declared for their canonical shard. Windows and macOS jobs run directly on the matching GitHub-hosted runner. `bootstrap-worker.sh` and `bootstrap-worker.ps1` install host prerequisites that are not supplied by those images.

## Compiler cache

For access to the existing Azure Blob sccache, define the optional repository or organization secret `AZURE_SCCACHE_CONNECTION_STRING`. The connection string is passed to the worker as `SCCACHE_AZURE_CONNECTION_STRING`; it is never written to the generated worker configuration. The cache uses container `releases` and key prefix `deeplearning4j/releases/compiler-cache/v1`.

When that secret is unavailable, such as on an untrusted fork, the existing GitHub Actions sccache setup is used as the fallback.

## Sonatype publication

The one dispatcher retains the historical snapshot-versus-release switch:

- `deployToReleaseStaging=0` builds `snapshotVersion`, merges the selected logical matrix once, and publishes it to `https://central.sonatype.com/repository/maven-snapshots/`.
- `deployToReleaseStaging=1` rewrites the same source snapshot to `releaseVersion`, builds and merges the selected logical matrix, signs the merged repository with `SONATYPE_GPG_KEY`, and uploads it to Central Portal as a user-managed deployment. It stops after validation for manual review; it never requests automatic publication.
- `dryRun=true` performs the complete build, merge, and verification without contacting Sonatype.

The release plumbing reuses `release/aws/build-platform.py` for version setup and `release/central/repository.py` for merge, signing, snapshot deployment, and release staging. A failed publication can be retried without rebuilding by setting `publishSourceRunId` to the original workflow run, retaining the same `deployToReleaseStaging` mode.

For example, stage one complete logical release matrix for manual Central review:

```bash
gh workflow run build-deploy-cross-platform.yml \
  --ref ag_new_release_updates_2 \
  -f workflow=build-deploy-linux-x86_64.yml \
  -f commitId=FULL_40_CHARACTER_SHA \
  -f deployToReleaseStaging=1 \
  -f releaseVersion=1.0.0-M3 \
  -f snapshotVersion=1.0.0-SNAPSHOT \
  -f dryRun=false
```

## Matrix maintenance

Add or change classifiers in the provider release plans first. Keep `release/github/workflow-matrix.json` limited to workflow-to-shard and shard-to-runtime mappings. `release/github/test_worker.py` rejects workflow rows that do not resolve to an explicit release-plan variant and checks the historically distinct Linux compile ISA classifiers.

A normal dispatch sets `workflow` to the desired logical matrix ID and runs that complete canonical matrix. Partial reruns must set `targetedRetry=1` and provide one or more exact published classifier IDs in `classifiers`; filters are rejected otherwise. This keeps recovery of a failed classifier explicit without allowing an intended complete release run to silently omit base or another variant.

ZLUDA is selected by its published CUDA/ROCm classifier, such as `linux-x86_64-cuda-12.9-zluda-rocm-7.2.4`. Published worker IDs and Maven classifiers use single hyphens and are the only supported classifier interface.

These workflows run the shared worker locally on GitHub runners; they do not provision AWS, Azure, or GCP virtual machines.
