# External AWS release builds

This directory moves the existing DL4J release compilation matrix off GitHub-hosted runners while leaving credentialed Maven Central publication in GitHub Actions.

The plan preserves the current release environments:

- Ubuntu 22.04 x86_64: CPU spins and CUDA 12.6/12.9 compilation.
- Ubuntu 22.04 ARM64 on Graviton: Linux ARM64 spins.
- Windows Server 2022: CPU and CUDA 12.6/12.9 compilation.
- macOS 14 ARM64 on EC2 Mac (`mac2-m2pro.metal`): base, compile, MPS and MPS+compile spins.
- Ubuntu 22.04/24.04 cross-compilers: Android ARM64 and x86_64 spins.
- Cross-platform Java, tokenizers, SDX Maven coordinates and SDK packaging are owned by the base platform executions.

Every compute host is CPU-only. CUDA builders install or enter the CUDA development toolchain but do not request a GPU/accelerator EC2 family. Both Maven-layout artifacts and SDX runtime/AOT SDK assets are required outputs where the existing workflows produce them.

The plan covers every current `build-deploy-*.yml` workflow, including cross-platform Rust/tokenizers and Java modules, compatibility artifacts, Vulkan/Vulkan-MLIR, Hexagon, TPU and ZLUDA. Linux, Windows and Android matrix variants expand to independent EC2 executions so native configurations build concurrently; the current plan expands 16 logical shards to 45 executions. macOS variants remain sequential on one EC2 Mac dedicated host to avoid allocating four separately billed 24-hour hosts.

Bootstrap parity is explicit rather than inherited from a hosted-runner image: Java 11, Maven, CMake/Ninja, LLVM/MLIR, native protoc 3.8.0 plus cross-platform protoc 21.7, Rust/cbindgen, OpenBLAS, Android NDK r27d (ARM64) and r26d (x86_64), platform compilers and CUDA tooling are installed by the workers. The embedded build driver preserves the release flags and uses per-shard `buildThreads` values sized for the EC2 host (10-48 threads), plus a platform-sized Maven heap. Rendered workers are staged in encrypted private S3 and fetched through short-lived presigned URLs, keeping EC2 user data below 16 KiB while allowing the project source commit to predate the orchestration code.

## Credentials and permissions

Install Python 3, `boto3`, and the GitHub CLI. The provisioner uses the standard boto3 credential chain; there is no custom credentials file or DL4J-specific AWS variable:

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...       # temporary credentials only
export AWS_REGION=us-east-1        # AWS_DEFAULT_REGION also works
```

AWS profiles, SSO, web identity and instance roles also work. The principal needs EC2 instance/network/image permissions, `AllocateHosts`/`ReleaseHosts` for EC2 Mac, IAM role/profile management, S3 bucket/object management, SSM parameter management and CloudWatch Logs group/stream management. `gh` authentication is needed only by `collect` when it creates the draft GitHub Release.

The default VPC is used unless `--subnet-id` and `--security-group-id` are supplied. EC2 Mac capacity is Availability-Zone-specific; choose a subnet in an AZ where `mac2-m2pro.metal` host capacity is available.

## Run the complete matrix

```bash
python3 -m pip install boto3

# Read-only: validates identity, every selected AMI's publisher/architecture/
# storage/virtualization, instance architecture, per-AZ offerings, default
# networking and the Standard On-Demand vCPU quota before creating resources.
python3 release/aws/release.py preflight

python3 release/aws/release.py start \
  --version 1.0.0-M3 \
  --snapshot-version 1.0.0-SNAPSHOT \
  --branch master \
  --reset-kill-switch

python3 release/aws/release.py status
```

`start` accepts exactly one of `--branch` or `--commit`. With `--branch`, the local provisioner resolves `refs/heads/<branch>` from `--repository` before making any AWS changes, then passes that immutable commit to every worker. A moving branch therefore cannot produce mixed revisions across the matrix. The output includes `sourceBranch` and `resolvedCommit`, along with the exact shutdown and live-log commands.

`start` repeats fail-closed matrix validation before it resets the kill switch, creates a bucket/log group/IAM profile, allocates a Mac host, or launches an instance. AMIs are discovered with EC2 `DescribeImages` rather than assuming that a public SSM parameter exists in every region. Ubuntu queries are restricted to Canonical owner `099720109477`; Windows and macOS queries are restricted to AWS-owned images. The returned image is then independently checked for owner (where a stable publisher account ID is available), architecture, `available` state, EBS root storage, HVM virtualization and Windows platform metadata. The selected instance types must exist, support the AMI architecture and be offered in the launch subnet's Availability Zone.

The image patterns intentionally mirror the hosted environments in the current Actions workflows: Ubuntu 22.04 (x86_64 and ARM64), Ubuntu 24.04 (Android x86_64), Windows Server 2022 Full Base and Amazon's macOS 14 ARM64 AMI. Image IDs remain regional and are resolved at invocation time. Canonical documents both its EC2 query pattern and publisher-account verification at <https://documentation.ubuntu.com/aws/en/latest/aws-how-to/instances/find-ubuntu-images/>; AWS documents regional AMI selection and owner/platform/root-device filtering at <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/finding-an-ami.html>.

Canonical's live EC2 image catalog is the final authority for release/volume combinations. As of July 2026, Jammy 22.04 uses the `hvm-ssd` name family in `us-east-1` (for example, catalog entries `ami-0d001f8052688dc45` for amd64 and `ami-09718fd66fac8c035` for arm64), while Noble 24.04 uses `hvm-ssd-gp3`. The plan follows those published families but still resolves the current regional ID on every invocation rather than pinning these examples.

To pin a commit explicitly instead:

```bash
python3 release/aws/release.py start \
  --version 1.0.0-M3 \
  --snapshot-version 1.0.0-SNAPSHOT \
  --commit "$COMMIT" \
  --reset-kill-switch
```

## Monitor live builds

Every execution streams bootstrap, dependency installation and compiler output into CloudWatch Logs group `/deeplearning4j/releases`, with one stream named `<RUN_ID>/<SHARD_ID>`. A heartbeat is emitted every 60 seconds even when a compiler produces no output. CloudWatch retains streams for 30 days by default; both settings are explicit in `release-plan.json`.

```bash
# EC2 state, shard identity, instance type, kill-switch state and log command:
python3 release/aws/release.py status --run-id "$RUN_ID"

# Follow all streams in the run. By default, follow starts with the last 10 minutes:
python3 release/aws/release.py logs --run-id "$RUN_ID" --follow

# Follow one logical shard (all its expanded variants) with a larger history window:
python3 release/aws/release.py logs --run-id "$RUN_ID" \
  --shard linux-x86_64-cpu --since-minutes 60 --follow

# Print all currently retained events once and exit:
python3 release/aws/release.py logs --run-id "$RUN_ID"
```

The final `build.log` is also uploaded to private S3 and included in the draft GitHub Release, so CloudWatch is the operational live view while S3/GitHub retain the release evidence.

CloudWatch log streams are independently deletable without touching artifacts or the log-group retention policy:

```bash
# Delete every stream belonging to one run:
python3 release/aws/release.py delete-logs --run-id "$RUN_ID" --yes

# Delete only one logical shard from that run:
python3 release/aws/release.py delete-logs --run-id "$RUN_ID" \
  --shard linux-x86_64-cpu --yes

# Explicitly delete every DL4J release log stream in this AWS region:
python3 release/aws/release.py delete-logs --all-runs --yes
```

Use `preflight --shard linux-x86_64-cpu--base` before a one-execution smoke build. A full fan-out intentionally needs a large Standard On-Demand vCPU quota; preflight prints the exact requirement for the selected executions and fails before provisioning when the account quota is insufficient.

Before starting any smoke or full build, keep this emergency command ready in a second terminal using the same standard AWS region environment:

```bash
python3 release/aws/release.py stop-everything --wait

# Stop compute and also delete every managed CloudWatch release log stream:
python3 release/aws/release.py stop-everything --wait --purge-logs

# Only when you also intend to delete every staged release object in the managed bucket:
python3 release/aws/release.py stop-everything --wait \
  --purge-storage --bucket dl4j-release-ACCOUNT-REGION
```

The first command is the normal shutdown path and preserves staged artifacts. It activates the kill switch before terminating tagged compute, so it is safe to invoke during provisioning or compilation. The second command is destructive and requires the exact managed bucket name.

Use repeated `--shard` options for a controlled partial build. Selecting a logical shard such as `linux-x86_64-cpu` launches all of its variants in parallel; selecting an expanded ID such as `linux-x86_64-cpu--avx2` launches only that execution. The complete release requires every expanded execution in `release-plan.json`.

Each worker checks out the exact commit, runs its assigned variant (or the consolidated macOS variants), uses Maven `install` rather than `deploy`, activates the SDX profile, stages only its owned Maven coordinates/classifiers, packages platform SDK JARs and SDX runtime/AOT assets, uploads checksum-addressed output to private S3, and terminates itself. Builders never receive GitHub, GPG, or Maven Central credentials.

Collect only after every planned shard succeeds:

```bash
python3 release/aws/release.py collect \
  --run-id 1.0.0-M3-0123456789 \
  --bucket dl4j-release-ACCOUNT-REGION \
  --release-tag deeplearning4j-1.0.0-M3-build \
  --version 1.0.0-M3 \
  --commit "$COMMIT"
```

The collector verifies status, shard identity, platform workload completeness and hashes before uploading every Maven/SDK archive and manifest to a draft GitHub Release. It also merges all Maven shards, rejects conflicting paths, validates every component, generates Maven checksums, and publishes an exploded Maven 2 repository at:

```text
s3://dl4j-release-ACCOUNT-REGION/deeplearning4j/releases/RUN_ID/maven-repository/
```

The repository is private. The collector uploads `.dl4j/complete.json` last; only use the layout when that marker exists and its `ready` value is `true`. `completeMatrix` distinguishes a full collection from a deliberately partial `--shard` collection. `.dl4j/repository-manifest.json` binds every repository file to its size and SHA-256 digest.

For local Maven testing, sync it with the normal AWS credential chain and use the downloaded directory as a `file://` repository:

```bash
aws s3 sync \
  s3://dl4j-release-ACCOUNT-REGION/deeplearning4j/releases/RUN_ID/maven-repository/ \
  ./dl4j-test-repository

test -f ./dl4j-test-repository/.dl4j/complete.json

mvn dependency:get \
  -Dartifact=org.nd4j:nd4j-native:VERSION:jar:linux-x86_64 \
  -DremoteRepositories=dl4j-aws-test::default::file://"$(pwd)"/dl4j-test-repository
```

The exploded S3 repository is testing output, not the Maven Central publication source. Central publication still consumes and re-verifies the immutable shard archives from the draft GitHub Release. `stop-everything --purge-storage` deletes the exploded repository, its completion metadata, archives and all bucket object versions.

## Stop everything

At any point, including provisioning and mid-compiler execution:

```bash
python3 release/aws/release.py stop-everything --wait
```

The command enables the global SSM kill switch first, then terminates every tagged Linux, Windows and macOS instance across all run IDs, cancels matching spot requests, and attempts to release every tagged EC2 Mac dedicated host. Workers poll the switch every 15 seconds and kill the active process tree. New starts check the switch before and immediately after provisioning each resource and require an explicit `--reset-kill-switch` afterward.

AWS imposes a minimum 24-hour allocation period on EC2 Mac dedicated hosts. An emergency stop terminates the Mac instance immediately, but AWS may reject `ReleaseHosts` until that period expires. Such hosts are reported under `pendingDedicatedHosts`; rerun `stop-everything --wait` after they become eligible. This AWS billing constraint cannot be bypassed by an API call.

Add `--purge-storage` to delete collected S3 objects as well, and `--purge-logs` to delete all streams in the configured release log group. Both deletions are separate opt-ins so release evidence is not destroyed accidentally. For run-scoped log cleanup, prefer `delete-logs --run-id ... --yes`.

## Maven Central handoff

After reviewing the draft GitHub Release, dispatch `.github/workflows/publish-central-from-release.yml`. The protected GitHub job uses the existing injected `GITHUB_TOKEN`, GPG key/passphrase and Central Portal token. It downloads every release asset, verifies version, commit, sizes and SHA-256 hashes, rejects conflicting Maven paths, and does not rebuild any platform artifact.

The workflow selects the publication protocol from the exact version:

- A non-SNAPSHOT version is signed, bundled, validated and uploaded through the Central Publisher Portal release API. The `automatic` input controls whether Central publishes immediately after validation.
- A version ending in `-SNAPSHOT` is deployed from the same verified prebuilt Maven repository with Maven Deploy Plugin 3.1.4 to `https://central.sonatype.com/repository/maven-snapshots/`. Maven creates Central-compatible timestamped snapshots and metadata; no native or Java compilation runs in GitHub Actions.

For a snapshot build, pass the snapshot as the build's final version, normally keeping the source snapshot version the same:

```bash
python3 release/aws/release.py start \
  --version 1.0.0-SNAPSHOT \
  --snapshot-version 1.0.0-SNAPSHOT \
  --branch master \
  --reset-kill-switch
```

Snapshot publishing must first be enabled for the DL4J namespace in Central Portal. Central currently retains published snapshots for 90 days. Release and snapshot publication remain manual protected GitHub environment actions.
