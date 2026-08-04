# Microsoft Azure DL4J release runner

This directory is the Azure sibling of `release/aws` and `release/gcp`. It runs the portable DL4J release matrix on Azure Virtual Machines while leaving signing and Maven Central publication in the protected GitHub Actions environment.

The controller keeps the established release contract intact:

- `release/aws/build-platform.py` remains the shared execution driver.
- `build-scripts/release/*.sh` remain the platform build entry points used by GitHub Actions.
- Compatibility lanes run concurrently; each VM drains a serial queue of matching shards while preserving its toolchains, Git objects, compiler cache, and durable per-shard Maven repositories.
- Workers produce the same Maven archives, SDK archives, status document, and checksummed shard manifest.
- `release/central/repository.py` materializes the test Maven repository and the existing Central workflow performs the final verified publication.

Every release VM uses pinned sccache 0.15.0 with a two-level local-disk and Azure Blob cache. Compiler objects are published directly under `deeplearning4j/releases/compiler-cache/v1` in the private artifact container, making them reusable by later Azure VMs and runs without moving a cache archive. The namespace is stable across branches, commits, and run IDs; sccache hashes the compiler, command, environment, and inputs. The controller signs a read/write/create-only container SAS that expires with the run and places only that short-lived SAS connection string in the worker payload—the storage account key never leaves the controller. Cache startup, per-variant statistics, and shutdown flow through the same live Blob logs as the build. Normal VM cleanup preserves the cache; `stop-everything --purge-storage` removes it because it is inside the managed artifact prefix.

## What Azure covers

The Azure plan includes every AWS lane that Azure can actually host:

- Ubuntu 22.04 x86_64 CPU, CUDA 12.6/12.9 compile, compatibility, Vulkan, Vulkan+MLIR, Hexagon, TPU compile, and ZLUDA.
- Ubuntu 22.04 ARM64 CPU on Azure Cobalt/Arm VM sizes.
- Windows Server 2022 x86_64 CPU and CUDA 12.6/12.9 compile.
- Ubuntu cross-builders for Android ARM64 and Android x86_64.
- Maven, tokenizers, Java/SDX coordinates, platform SDKs, and runtime/AOT assets owned by those lanes.

Every Azure builder is CPU-only. CUDA, TPU, Vulkan, Hexagon, and ZLUDA lanes compile their toolchains and classifiers but do not request GPU, TPU, or other accelerator VM families. The Linux ZLUDA lane layers the pinned ROCm 7.2.4 HIP development runtime and MIOpen development package into its disposable CUDA 12.9 container. It verifies the SDK headers and shared libraries at build time but deliberately skips GPU discovery and never installs an AMD kernel driver; AMD hardware execution belongs in a separate opt-in validation run.

The full plan resolves to four independently schedulable Azure VMs:

- `linux-x86-64-jammy` handles the Ubuntu 22.04 x64 native, CUDA, Android ARM64, compatibility, Vulkan, Hexagon, TPU, and ZLUDA shards.
- `linux-x86-64-noble` handles the Ubuntu 24.04 Android x86_64 shard without weakening its image contract.
- `linux-arm64-jammy` handles the native Arm64 shard.
- `windows-x86-64-2022` handles the Windows CPU and CUDA shards.

Selecting fewer shards creates only the lanes that contain them. Azure has no macOS VM lane; use the AWS macOS builder for that part of a hybrid release.

Azure Virtual Machines do not provide the macOS 14 ARM64/MPS lane. `release-plan.json` marks `build-deploy-mac-arm64.yml` unsupported. Azure-only collection therefore reports `completeMatrix: false`. Collect the AWS EC2 Mac shard to the draft GitHub Release first, then run the Azure collector against the same immutable tag/version/commit to form the complete hybrid matrix.

Azure also has no Google Cloud TPU hardware equivalent, so there is deliberately no Azure `tpu-smoke` command. The portable TPU classifier remains a compile-only x86 lane.

## Authentication and configuration

Use Python 3.10 or newer:

```bash
python3 -m pip install -r release/azure/requirements.txt
```

The controller uses `DefaultAzureCredential`. Normal Azure CLI credentials, a managed identity, workload identity, and service-principal environment variables are supported. Select the subscription and location with standard environment variables:

```bash
az login
export AZURE_SUBSCRIPTION_ID=00000000-0000-0000-0000-000000000000
export AZURE_LOCATION=eastus2
```

For a service principal, set `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, and `AZURE_CLIENT_SECRET` in the normal Azure SDK manner. Validate the resolved identity, subscription, and location with:

```bash
python3 release/azure/release.py configure
```

If credentials or required values are absent in an interactive terminal, the controller asks once for `az login`, `AZURE_SUBSCRIPTION_ID`, and `AZURE_LOCATION`. Redirected and CI invocations fail immediately. Put `--no-wizard` before the subcommand to force noninteractive behavior.

The controller principal needs permission to create and delete resource groups/resources, virtual machines, managed identities, role assignments, networks, public IPs, and storage accounts, and to read VM-family quota. A subscription-level Owner role is sufficient for initial setup; a narrower custom role can be used if it includes those operations and role-assignment rights.

Workers use a user-assigned managed identity scoped to the managed storage account with the built-in Storage Blob Data Contributor role. Storage keys, GitHub credentials, GPG keys, and Central credentials are never placed on builder VMs.

## Preflight

`preflight` performs live subscription checks before provisioning:

- Resolves every selected Azure VM resource SKU in the requested location.
- Verifies x64 versus Arm64 CPU architecture, availability-zone restrictions, vCPU count, and memory.
- Resolves every Marketplace image publisher/offer/SKU/version.
- Reads current total-regional and per-family vCPU quota and validates the sum of every simultaneously running lane against both limits.
- Searches viable size combinations jointly. It first maximizes the useful minimum lane size, then aggregate throughput and balance, so one box is not starved merely because another candidate appears earlier.
- Validates the storage-account name and the 64-4095 GiB OS-disk bound.

```bash
python3 release/azure/release.py preflight \
  --shard linux-arm64-cpu--base \
  --max-cores 32
```

The default candidates are compute-optimized Fsv2 and F-as-v7 x64 VMs plus Dpsv6/Dpsv5 Arm64 VMs. The preferred order is interleaved by core count, and preflight skips candidates whose current family quota cannot admit the VM. Candidate availability is never assumed from the JSON file; the live Resource SKUs and usage APIs are authoritative. Force a verified size or availability zone only when needed:

```bash
python3 release/azure/release.py preflight \
  --lane-machine linux-x86-64-jammy=Standard_F32as_v7 \
  --lane-machine windows-x86-64-2022=Standard_F16as_v7 \
  --zone 1
```

Use `--max-cores` as a per-VM ceiling and `--max-total-cores` as an aggregate cost/concurrency ceiling. `--machine-type` still forces one size for every selected lane and is mainly useful for same-architecture selections; repeated `--lane-machine LANE=SIZE` overrides mixed combinations safely. Build threads default to `min(plan threads, vCPUs / 2)`, and Maven heap is reduced to fit each selected VM.

## Start and monitor

A one-variant smoke build:

```bash
python3 release/azure/release.py start \
  --branch ag_new_release_updates_2 \
  --version 1.0.0-SNAPSHOT \
  --snapshot-version 1.0.0-SNAPSHOT \
  --shard linux-x86_64-cpu--base \
  --max-cores 32 \
  --reset-kill-switch
```

A full Azure-capable matrix:

```bash
python3 release/azure/release.py start \
  --branch ag_new_release_updates_2 \
  --version 1.0.0-SNAPSHOT \
  --snapshot-version 1.0.0-SNAPSHOT \
  --max-cores 72 \
  --max-total-cores 192 \
  --reset-kill-switch
```

`start` resolves a branch to one immutable 40-character commit before creating resources. Creating or locating the resource group and storage account is the unavoidable, idempotent bootstrap needed to host the lock; immediately afterward a renewable, crash-expiring lease on a run-scoped kill-switch Blob serializes that run's state, identity, network, and VM mutations while allowing independent run IDs to proceed concurrently. Every controller gets a unique epoch. Kill-switch writes must present the Blob lease, run-manifest updates use epoch-checked ETag compare-and-swap, and run-scoped identities/VMs/NICs/public IPs carry the epoch and use epoch-specific names. The controller writes an `initializing` audit manifest before disabling the kill switch, then creates a virtual network/NSG, a run-scoped user-assigned managed identity, and every selected compatibility-lane VM concurrently. A short-lived read-only SAS URL bootstraps each rendered worker; all subsequent log, kill-switch, status, and artifact traffic uses managed identity. Fresh role assignments receive a bounded propagation window before workers fail closed. Lane events are folded into `run.json` by one controller path, completed shards remain successful even if a later shard in their lane fails, and the first failed lane signals sibling tasks locally and enables only that run's kill switch while every lane performs dependency-ordered cleanup. Azure long-running operations are fenced before submission and after completion; the local cancellation signal also interrupts siblings that are still provisioning. After lease loss, the stale controller cannot write the leased switch, cannot overwrite a newer manifest ETag, and performs no further shared mutation. The run identity is deleted only after all lane futures finish and no run-tagged VM remains.

Linux installs a systemd service from cloud-init; Windows uses the Azure Custom Script Extension only to persist the rendered worker, register a SYSTEM startup task, and wait for a durable startup marker. Both supervisors survive a reboot. A lane keeps one bare Git object store, one toolchain directory, and one ccache/sccache directory for its entire VM lifetime. Every shard gets a durable but isolated Maven repository, fresh source tree, and isolated output directory, preventing a prior shard's Maven artifacts or metadata from entering a later archive. Before accepting a checkpoint after reboot, both the worker and controller validate its controller epoch, run/repository/commit/version identity, exact variant list, and a canonical SHA-256 digest of the complete shard contract (workloads, platform/backend, build properties, image, and toolchain/container choices included). Successful matching shards are skipped, append-only transcripts and compiler/toolchain caches are retained, and only incomplete source/output state is rebuilt. Kill handling terminates the active process tree and lets final status/log publication finish before shutdown, with a bounded forced-poweroff fallback. The VM shuts down only after its complete queue is terminal. Each VM has an explicit outbound public IP but the managed NSG contains no inbound allow rules.

Both preflight output and `run.json` use schema version 1. `run.json` retains the flat `executions` array used by status, collection, and historical tooling and adds a `lanes` array plus `laneId` on each execution so selected machine combinations, VM resources, queue membership, and lane cleanup are auditable without changing per-shard artifact paths.

Parallelism overlaps VM and Premium SSD exposure. Use `--max-cores`, `--max-total-cores`, or a smaller shard selection when you want a tighter cost envelope; no idle VM pool is retained after the run.

The controller prints recovery commands before the first lane starts:

```bash
python3 release/azure/release.py status --run-id RUN_ID
python3 release/azure/release.py logs --run-id RUN_ID --follow
python3 release/azure/release.py stop-everything --wait
```

Workers append only new output to private `live.log` Append Blobs while the lane runs. The default `logs --follow` view reads the cumulative lane transcript, so VM bootstrap, toolchain installation, and every queued shard remain visible from one command on both Linux and Windows. Add `--shard SHARD_ID` to drill into that shard's live transcript; after it is terminal the same command reads its immutable `build.log`. Every append carries Azure's expected-position condition and ambiguous responses are reconciled against the remote prefix before the local offset advances, preventing retry duplication. A failed shard's retained `build.log` is printed automatically by the attached controller before it raises the failure.

`status` includes the run manifest plus each active VM's size, provisioning state, power state, complete Azure instance-view statuses, and a bounded managed-boot-diagnostics console tail. Boot diagnostics are enabled on every created VM, providing the same bootstrap-failure visibility that the AWS controller obtains from EC2 instance health and console output.

Output for each shard is stored beneath:

```text
https://ACCOUNT.blob.core.windows.net/releases/deeplearning4j/releases/RUN_ID/SHARD/
```

## Emergency stop and cleanup

The control container holds a fail-closed global emergency switch plus one leased kill-switch document per run. A run controller leases only its own document, so Windows, Linux, and Android releases can coexist; ordinary lane failure or completion affects only sibling lanes in that run. Workers poll both documents every 15 seconds before and during toolchain installation/build execution. Missing, malformed, unreadable, or unexpectedly epoch-mismatched run state stops the worker. The global document is acted on only when it carries an emergency `force` record, so a legacy controller's normal completion record cannot cancel newer independent runs.

```bash
python3 release/azure/release.py stop-everything --wait
```

The command looks up existing storage without creating anything, breaks (or proves the absence of) the global emergency-switch lease, immediately acquires it, writes a forced stop record, and then fences every visible run lease before deleting anything. If it loses the acquisition race or cannot keep renewing the fence, it aborts resource deletion. The emergency command holds the lease until cleanup finishes, so even a concurrent `start --reset-kill-switch` cannot provision replacements mid-cleanup. Once fenced, it deletes tagged Azure VMs and waits, deletes dependent network interfaces and waits again, and finally deletes public IPs plus run-scoped managed identities and role assignments. Storage and retained release evidence are preserved by default.

For a completed run with no remaining VM, delete only its retained Blob logs with:

```bash
python3 release/azure/release.py delete-logs --run-id RUN_ID --yes
```

`delete-logs` acquires and continuously checks the same controller lease, validates every target before deleting anything, and rejects non-terminal runs or runs that still have an Azure VM. During an emergency cleanup, delete retained Blob log objects with:

```bash
python3 release/azure/release.py stop-everything --wait --purge-logs
```

Delete all staged release objects only when explicitly intended:

```bash
python3 release/azure/release.py stop-everything --wait --purge-storage
```

The control kill switch is recreated/enabled after a storage purge so old workers cannot become launchable.

## Collect, hybrid completion, and Central

A successful `start` or `resume` automatically performs a Blob-only collection after VM and identity cleanup. Raw shard archives remain under their shard paths as immutable provenance, while the usable Maven 2 repository is materialized with normal Maven coordinates at:

```text
deeplearning4j/releases/RUN_ID/maven-repository/
```

Repository files are uploaded as individual blobs, followed by `.dl4j/repository-manifest.json` and its SHA-256 file; `.dl4j/complete.json` is written last. The run root also receives `release-build-manifest.json` and its checksum, so consumers do not need to inspect the raw `.tar.gz` files. Pass `--no-auto-collect` only when deliberately retaining raw outputs without publishing the expanded repository.

Run `collect` explicitly when assembling a hybrid release or updating its draft GitHub Release:

```bash
python3 release/azure/release.py collect \
  --run-id RUN_ID \
  --release-tag dl4j-1.0.0-external \
  --version 1.0.0 \
  --commit FULL_40_CHARACTER_SHA
```

To publish only the expanded Blob repository for an existing successful run, without downloading the SDK archive or touching GitHub, add `--no-github --repository-only`. This is the same lightweight mode used by automatic collection.

`collect` checks `run.json`, every `status.json`, every shard identity, release version, commit, and workload/variant selection. Every archive included in a full or GitHub collection is verified against the worker-attested size and SHA-256 digest before the expanded repository is replaced; repository-only mode deliberately omits SDK archives from its manifest instead of claiming an unverified digest.

Azure-only output remains intentionally incomplete because the macOS/MPS variants are absent. For hybrid assembly, collect the AWS macOS lane first and then collect the Azure lanes with the same draft tag, version, and commit; cross-provider collectors must run sequentially. The Azure collector holds the same renewable controller lease while collecting, requires an existing release manifest to download and parse successfully, downloads and hash/size-verifies every retained GitHub asset, and derives retained coverage only from verified shard manifests plus their required Maven/SDK archives. New manifests attest `variants` directly; for legacy AWS manifests, exact variants are inferred from hash-verified classifier files in the worker shard manifest rather than assuming that a parent lane built everything. It re-fetches the GitHub manifest immediately before replacement; a changed manifest aborts collection so the operator can rerun against the new state. New data assets are uploaded without `--clobber`, while duplicate content keeps normalized provider/source provenance and every other asset field must agree. Completeness is calculated from exact attested classifier variants and emitted as `matrixEntries`, so a parent lane with exclusions cannot claim its omitted variants.

Only dispatch `.github/workflows/publish-central-from-release.yml` after `release-build-manifest.json` reports `completeMatrix: true` and empty missing lists. That protected workflow re-verifies the immutable artifacts, signs releases, and publishes either Central Portal releases or Central-compatible snapshots. Azure builders never receive publication credentials.

## Azure references

- Arm64 Cobalt VM sizes: <https://learn.microsoft.com/azure/virtual-machines/sizes/general-purpose/dpsv6-series>
- x64 Fsv2 VM sizes: <https://learn.microsoft.com/azure/virtual-machines/sizes/compute-optimized/fsv2-series>
- Managed identities for Azure resources: <https://learn.microsoft.com/entra/identity/managed-identities-azure-resources/overview>
- Azure Blob authorization with managed identity: <https://learn.microsoft.com/azure/storage/blobs/authorize-access-azure-active-directory>
