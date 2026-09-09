# GitHub release worker

`build-deploy-cross-platform.yml` is the single registered GitHub release dispatcher. Its `workflow` input selects a logical matrix ID from `release/github/workflow-matrix.json`; those IDs retain the historical `build-deploy-*.yml` names but do not require matching physical workflow files. The dispatcher calls `.github/workflows/_release-worker.yml`, which obtains its classifiers from `release/aws/release-plan.json` through the matrix and executes the same `release/aws/build-platform.py` worker used by the cloud release controllers.

Linux jobs run in the container image declared for their canonical shard. Windows and macOS jobs run directly on the matching GitHub-hosted runner. `bootstrap-worker.sh` and `bootstrap-worker.ps1` install host prerequisites that are not supplied by those images.

## Compiler cache

For access to the existing Azure Blob sccache, define the optional repository or organization secret `AZURE_SCCACHE_CONNECTION_STRING`. The connection string is passed to the worker as `SCCACHE_AZURE_CONNECTION_STRING`; it is never written to the generated worker configuration. The cache uses container `releases` and key prefix `deeplearning4j/releases/compiler-cache/v1`.

When that secret is unavailable, such as on an untrusted fork, the existing GitHub Actions sccache setup is used as the fallback.

## Existing Azure cache migration to R2

`migrate-cache-to-r2.yml` is a manual, GitHub-hosted transfer, not a native build.
It requires `AZURE_SCCACHE_CONNECTION_STRING`, `R2_ACCESS_KEY_ID`, and
`R2_SECRET_ACCESS_KEY`. The private destination is bucket `dl4j-cache` at
`https://318204901782458555a243ad96f80e3f.r2.cloudflarestorage.com`.

Run `preview` first to check credentials and inventory the compiler, toolchain,
and dependency cache namespaces. Then run `copy`; it preserves object paths,
copies missing/changed objects, and verifies source/destination bytes with
`rclone check --download --one-way`. `verify` repeats just the byte check.
Empty source namespaces fail rather than being reported as successful migrations.
No mode deletes Azure objects or extra R2 objects. Failed/interrupted copies can
be rerun; no cache clearing or recompilation is needed.

Copying and byte verification both read Azure data and can incur Azure egress
charges. Source writers must be quiesced for the final delta and verification;
a live copy is not an atomic snapshot. Inspect the Actions inventory and final
verification result, not just the transfer progress. Use the already-registered dispatcher on the working branch:

```bash
gh workflow run build-deploy-cross-platform.yml --ref ag_new_release_updates_2 \
  -f cacheMigration=preview
```

After reviewing the preview, use `cacheMigration=copy`. The migration option
skips all build and publication jobs; ordinary dispatches remain unchanged.

This is the **data-copy stage only**: it leaves existing build workers and Azure
support unchanged. Backend selection, S3-enabled sccache binaries, R2 endpoint
configuration, archive transports, and any Azure URLs embedded in manifests
must be qualified separately before switching workers. A verified copy alone
does not establish compiler-cache hits or complete the cutover.

## Sonatype publication

The one dispatcher retains the historical snapshot-versus-release switch:

- `deployToReleaseStaging=0` builds `snapshotVersion`, merges the selected logical matrix once, and publishes it to `https://central.sonatype.com/repository/maven-snapshots/`.
- `deployToReleaseStaging=1` rewrites the same source snapshot to `releaseVersion`, builds and merges the selected logical matrix, signs the merged repository with `GPG_PRIVATE_KEY`, and uploads it to Central Portal as a user-managed deployment. It stops after validation for manual review; it never requests automatic publication.
- `dryRun=true` builds and verifies a repository without signing or uploading to Sonatype. The repository is retained as an Actions artifact for seven days, including hidden directory contents. Use `workflow=all` for full Maven repository assembly. The dispatcher's default matrix still contains only four CPU base variants and is not a full release.

The release plumbing reuses `release/aws/build-platform.py` for version setup and `release/central/repository.py` for merge, signing, snapshot deployment, and release staging. `publishSourceRunId` recovers selected worker artifacts, not a full assembled repository: it does not reconstruct the Java reactor or apply full-matrix ownership. For a failed full assembly, rerun failed jobs in the original Actions run so the successful worker artifacts remain available. Publication tooling comes from the workflow revision; worker manifests must still match the immutable build source SHA and version.

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

`workflow=all` derives every native variant from the canonical release plan (currently 66 variants), rejects partial classifier filters, and runs the Python release contract suites before any native jobs start. A successful contract check is not a successful native build.

After all workers succeed, `full-repository.py`:

1. Requires exactly the canonical worker set, one immutable source SHA/version/run, matching plan digests, successful completion receipts, and attested classifier archives. Successful artifacts from earlier attempts of the same run may be reused.
2. Selects one declared owner for each shared native component. CPU base Linux owns CPU common, tokenizers, and SDX components; each CUDA version owns distinct backend/common coordinates such as `nd4j-cuda-12.9-backend-common`. It never resolves release conflicts by taking the first duplicate.
3. Builds the default Java reactor and its parent POMs from the same source in a fresh Maven repository seeded only with the attested native outputs. It then builds native, Vulkan, ZLUDA, and all three CUDA platform modules. Non-SNAPSHOT components must supply their generated sources and javadoc attachments.
4. Resolves each shipped binary independently using its installed POM, rejecting any DL4J dependency obtained remotely rather than produced by this run. This checks consumer resolution on the Linux assembly host; it does not certify every optional dependency or OS-activated Maven profile.
5. Emits `repository-manifest.json` with per-file owners, sizes, and SHA-256 hashes, verifies release metadata, and retains the repository under `merged-maven-repository-<version>/maven-repository/org/eclipse/deeplearning4j/`. Contract logs, effective reactor POMs, dependency-resolution evidence, and the manifest are retained for 90 days.

The full Maven run does not claim to package optional Spark/profile-only reactors, standalone SDK archives, or optional SDX AOT outputs. SDK payloads are excluded from full-run worker uploads to avoid duplicating them in the Maven assembly job.

Build-only qualification command (no signing, upload, tags, or automatic release):

```bash
gh workflow run build-deploy-cross-platform.yml \
  --ref ag_new_release_updates_2 \
  -f workflow=all \
  -f commitId=FULL_40_CHARACTER_SHA \
  -f deployToReleaseStaging=1 \
  -f releaseVersion=1.0.0-M3 \
  -f snapshotVersion=1.0.0-SNAPSHOT \
  -f dryRun=true
```

Release approval still requires a completed full run and inspection of its manifest, signing/key validity, Central namespace authorization, and measurement of the signed ZIP against Central's 1 GB bundle limit. The existing publisher rejects an oversized single bundle; component-preserving partitioning must be qualified if the assembled release exceeds that limit. A dry run proves none of the credential/Portal checks. Only an explicitly approved subsequent staging operation may upload a user-managed deployment; Central `VALIDATED` and human review are required before publication.

## Matrix maintenance

Add or change classifiers in the provider release plans first. Keep `release/github/workflow-matrix.json` limited to workflow-to-shard and shard-to-runtime mappings. `release/github/test_worker.py` rejects workflow rows that do not resolve to an explicit release-plan variant and checks the historically distinct Linux compile ISA classifiers.

A normal dispatch sets `workflow` to the desired logical matrix ID and runs that complete canonical matrix. Partial reruns must set `targetedRetry=1` and provide one or more exact published classifier IDs in `classifiers`; filters are rejected otherwise. This keeps recovery of a failed classifier explicit without allowing an intended complete release run to silently omit base or another variant.

ZLUDA is selected by its published CUDA/ROCm classifier, such as `linux-x86_64-cuda-12.9-zluda-rocm-7.2.4`. Published worker IDs and Maven classifiers use single hyphens and are the only supported classifier interface.

These workflows run the shared worker locally on GitHub runners; they do not provision AWS, Azure, or GCP virtual machines.
