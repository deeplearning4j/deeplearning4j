# Google Cloud DL4J release runner

This is the Google Cloud sibling of `release/aws`. It keeps the proven part of that backend intact: the same `release/aws/build-platform.py` execution driver, the same shared `build-scripts/release/*.sh` platform entry points used by GitHub Actions, the same Maven/SDK workloads, and every classifier variant in `release-plan.json`.

The controller launches one serial lane at a time. All variants in a lane run on the same VM, so ccache/sccache and the local Maven repository are reused. `--shard linux-x86_64-cpu--base` runs only the base variant; `--shard linux-x86_64-cpu` runs all seven CPU variants.

## Boundary: macOS and TPU hardware

Compute Engine does not offer macOS hosts. The GCP plan therefore covers all Linux, Windows, Android, CUDA compile, TPU compile, Vulkan, Hexagon, and ZLUDA lanes, but explicitly marks `build-deploy-mac-arm64.yml` unavailable. Use the AWS EC2 Mac lane for that artifact and assemble a hybrid release. GCP-only release manifests have `completeMatrix: false` so they cannot be mistaken for a complete DL4J Central release.

The `linux-x86_64-tpu` build lane is compile-only and runs on an ordinary C4 VM, following the same profile-gated compile-only rule as CUDA, Vulkan, Hexagon, and ZLUDA. Metal/MPS remains part of the unavailable macOS lane. The TPU C API is vendored and `libtpu.so` is loaded at runtime; actual TPU allocation happens only with the explicit `tpu-smoke` command.

## Authentication and configuration

Use Python 3.10 or newer and install the controller libraries once:

```bash
python3 -m pip install -r release/gcp/requirements.txt
```

The controller uses Google Application Default Credentials. A service-account key uses only the normal environment variables:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json
export GOOGLE_CLOUD_PROJECT=my-project-id
export GOOGLE_CLOUD_REGION=us-central1
```

For local user credentials, `gcloud auth application-default login` is also supported. The ADC search order is documented at <https://cloud.google.com/docs/authentication/application-default-credentials>.

Enable these APIs in the project:

```bash
gcloud services enable compute.googleapis.com storage.googleapis.com logging.googleapis.com tpu.googleapis.com cloudquotas.googleapis.com
```

The controller identity needs permission to create/delete Compute Engine instances, attach the worker service account, administer the managed Storage bucket, write/delete Cloud Logging logs, and create/delete TPU nodes. Typical predefined roles are:

- `roles/compute.instanceAdmin.v1`
- `roles/iam.serviceAccountUser`
- `roles/storage.admin`
- `roles/logging.admin`
- `roles/cloudquotas.viewer`
- `roles/tpu.admin`
- `roles/serviceusage.serviceUsageConsumer`

Pass the worker identity with `--service-account`; otherwise Google Cloud's default Compute service account is attached. The controller grants that identity `roles/storage.objectAdmin` only on the run's dedicated managed bucket and `roles/storage.objectViewer` on the project-wide control bucket. The identity must already have project-level `roles/logging.logWriter` (Cloud Logging failure is retained in the GCS/serial diagnostics rather than silently losing the build log).

## Preflight

Preflight does live API resolution. It does not trust names in the JSON plan: it resolves each official image family, verifies its architecture, verifies every selected machine type in an UP zone in the chosen region, reads current `CPUS_PER_VM_FAMILY` limits through the Cloud Quotas API, calculates the serial peak, and fails before provisioning if the selected constraint cannot run. Google exposes the effective family limit but not live allocation usage through that API; final launch admission still accounts for other VMs already consuming the quota.

```bash
python3 release/gcp/release.py preflight \
  --shard linux-x86_64-cpu--base \
  --max-cores 96
```

The plan uses the official `ubuntu-os-cloud` (`ubuntu-2204-lts`, `ubuntu-2204-lts-arm64`, `ubuntu-2404-lts-amd64`) and `windows-cloud` (`windows-2022`) families. It greedily tries verified C4 `c4-highcpu-*` and Arm C4A `c4a-highcpu-*` sizes. At launch time it retries every verified zone and then smaller verified machine sizes when Google reports a capacity, stockout, or quota admission failure; every fallback still obeys `--max-cores` and the effective family quota. C4/C4A do not support Persistent Disk and require gVNIC, so the controller explicitly provisions a `hyperdisk-balanced` boot disk and `GVNIC` interface. See <https://cloud.google.com/compute/docs/general-purpose-machines> and <https://cloud.google.com/compute/docs/images/os-details>.

To cap cost/quota at 16 cores, for example:

```bash
python3 release/gcp/release.py preflight --max-cores 16
```

The chosen build thread count is `min(plan threads, vCPUs / 2)` unless `--build-threads` is specified. Heap is also recalculated from machine memory. A forced type or zone is still validated:

```bash
python3 release/gcp/release.py preflight \
  --machine-type c4-highcpu-48 \
  --zone us-central1-a
```

## Start and monitor

A one-variant smoke build:

```bash
python3 release/gcp/release.py start \
  --branch ag_new_release_updates_2 \
  --version 1.0.0-SNAPSHOT \
  --snapshot-version 1.0.0-SNAPSHOT \
  --shard linux-x86_64-cpu--base \
  --max-cores 96 \
  --reset-kill-switch
```

A full GCP-capable matrix:

```bash
python3 release/gcp/release.py start \
  --branch ag_new_release_updates_2 \
  --version 1.0.0-SNAPSHOT \
  --snapshot-version 1.0.0-SNAPSHOT \
  --max-cores 96 \
  --reset-kill-switch
```

Everything except the already-built CPU base variant:

```bash
python3 release/gcp/release.py start \
  --branch ag_new_release_updates_2 \
  --version 1.0.0-SNAPSHOT \
  --snapshot-version 1.0.0-SNAPSHOT \
  --exclude-shard linux-x86_64-cpu--base \
  --max-cores 96 \
  --reset-kill-switch
```

The controller prints `status`, `logs`, and emergency shutdown commands as soon as it creates the run. It continuously emits provisioning state changes, serial-console output, Cloud Logging entries, and a 30-second controller heartbeat. It also retrieves the retained GCS build log when bootstrap or the build fails.

```bash
python3 release/gcp/release.py status --run-id RUN_ID
python3 release/gcp/release.py logs --run-id RUN_ID --follow
```

Linux startup uses the documented `startup-script` metadata key and Windows uses `windows-startup-script-ps1`; see <https://cloud.google.com/compute/docs/instances/startup-scripts/linux> and <https://cloud.google.com/compute/docs/instances/startup-scripts/windows>.

## Stop everything and purge logs

Every worker polls the project-wide `dl4j-release-PROJECT-control` bucket's kill-switch object every 15 seconds, so a stop in one region also blocks launches in a newly used region until an explicit `--reset-kill-switch`. Linux and TPU workers begin polling before package installation; Windows begins as soon as its dependency-free Python transport is installed. An unreadable or missing kill switch fails closed and shuts the worker down. The controller can also delete the VMs directly, so this works during package installation, checkout, or a native build:

```bash
python3 release/gcp/release.py stop-everything --wait
```

It first enables the project-wide control switch, requests deletion of every label-managed Compute Engine instance and TPU node, and then signals compatibility switches in every bucket carrying the controller's managed-bucket label. TPU discovery consumes every page of Cloud TPU service locations. Bucket-name prefixes alone are never sufficient for deletion or purge, so unrelated resources are not targeted; Storage signaling failures also cannot prevent the direct Compute/TPU deletion attempts.

Purge Cloud Logging logs and retained GCS `build.log`/`tpu-smoke.log` objects:

```bash
python3 release/gcp/release.py stop-everything --wait --purge-logs
```

Or purge logs for one completed run:

```bash
python3 release/gcp/release.py delete-logs --run-id RUN_ID --yes
```

Delete every object in the dedicated managed release bucket only when that is intended:

```bash
python3 release/gcp/release.py stop-everything --wait --purge-logs --purge-storage
```

The managed bucket has versioning disabled and its soft-delete retention set to zero, so purge is an actual deletion rather than a hidden retained generation. A storage purge preserves/recreates the enabled global kill-switch control object, preventing a stopped workload from becoming launchable just because its logs and artifacts were deleted. Disabling soft delete follows <https://cloud.google.com/storage/docs/disable-soft-delete>.

## Real TPU smoke test

Validate the configured TPU type/runtime before paying for a node:

```bash
python3 release/gcp/release.py preflight \
  --include-tpu-smoke \
  --tpu-zone us-central1-a
```

Create a single-chip v5e TPU VM, run the exact `TpuBackendSmokeTest` sequence from `.github/workflows/run-tpu-smoke-tests.yml`, upload logs/reports, and delete the node in `finally`:

```bash
python3 release/gcp/release.py tpu-smoke \
  --branch ag_new_release_updates_2 \
  --tpu-zone us-central1-a \
  --accelerator-type v5litepod-1 \
  --runtime-version v2-alpha-tpuv5-lite \
  --reset-kill-switch
```

`--spot` requests a preemptible TPU node. The default `v5litepod-1` maps to the documented one-chip v5e VM, and `v2-alpha-tpuv5-lite` is Google's current common v5e software version. Both are checked with the live API before creation. Current configurations, runtimes, and zones are documented at <https://cloud.google.com/tpu/docs/v5e>, <https://cloud.google.com/tpu/docs/runtimes>, and <https://cloud.google.com/tpu/docs/regions-zones>.

## Artifacts, Maven layout, and GitHub releases

Each lane writes to:

```text
gs://dl4j-release-PROJECT-REGION/deeplearning4j/releases/RUN_ID/SHARD/
```

It includes `maven-repository.tar.gz`, `sdk-assets.tar.gz` when applicable, `shard-manifest.json`, `status.json`, and `build.log`. `collect` validates status/identity, materializes the repository through `release/central/repository.py`, and publishes a valid Maven 2 layout at:

```text
gs://BUCKET/deeplearning4j/releases/RUN_ID/maven2/
```

A `.dl4j/complete.json` marker is written last. `collect` also creates or updates the draft GitHub release with the existing injected `gh` credentials; use `--no-github` for a GCS-only collection:

```bash
python3 release/gcp/release.py collect \
  --run-id RUN_ID \
  --release-tag dl4j-1.0.0-external \
  --version 1.0.0 \
  --commit FULL_40_CHARACTER_SHA
```

Because GCP cannot produce the macOS lane, a GCP-only manifest is deliberately marked incomplete for Central. Hybrid assembly is supported on the same draft tag: collect the AWS lanes first (for example, everything except TPU), then run the GCP `collect` command with that same `--release-tag`, version, and commit. The GCP collector downloads the existing manifest, rejects any identity mismatch, preserves its assets, overlays the GCP assets, and computes completeness at the individual classifier-variant level. Run `.github/workflows/publish-central-from-release.yml` only after the resulting manifest reports `completeMatrix: true`; the workflow rejects an explicitly incomplete manifest.
