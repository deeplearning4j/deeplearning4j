# GitHub release worker

The `build-deploy-*.yml` workflows are thin callers of `.github/workflows/_release-worker.yml`. The reusable workflow obtains its classifiers from `release/aws/release-plan.json` through `release/github/workflow-matrix.json`, then executes the same `release/aws/build-platform.py` worker used by the cloud release controllers.

Linux jobs run in the container image declared for their canonical shard. Windows and macOS jobs run directly on the matching GitHub-hosted runner. `bootstrap-worker.sh` and `bootstrap-worker.ps1` install host prerequisites that are not supplied by those images.

## Compiler cache

For access to the existing Azure Blob sccache, define the optional repository or organization secret `AZURE_SCCACHE_CONNECTION_STRING`. The connection string is passed to the worker as `SCCACHE_AZURE_CONNECTION_STRING`; it is never written to the generated worker configuration. The cache uses container `releases` and key prefix `deeplearning4j/releases/compiler-cache/v1`.

When that secret is unavailable, such as on an untrusted fork, the existing GitHub Actions sccache setup is used as the fallback.

## Sonatype snapshots

Every successful worker publishes its verified staged Maven coordinates to the Central Portal snapshots repository at `https://central.sonatype.com/repository/maven-snapshots/`. The reusable workflow requires the existing `CENTRAL_SONATYPE_TOKEN_USERNAME` and `CENTRAL_SONATYPE_TOKEN_PASSWORD` secrets, configures Maven server ID `central-portal-snapshots`, and invokes `release/central/repository.py deploy-snapshot` against the prebuilt staging tree. Publication does not rebuild the module, and a failed build still retains its worker artifact without attempting a deploy.

## Matrix maintenance

Add or change classifiers in the provider release plans first. Keep `release/github/workflow-matrix.json` limited to workflow-to-shard and shard-to-runtime mappings. `release/github/test_worker.py` rejects workflow rows that do not resolve to an explicit release-plan variant and checks the historically distinct Linux compile ISA classifiers.

These workflows run the shared worker locally on GitHub runners; they do not provision AWS, Azure, or GCP virtual machines.
