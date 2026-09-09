# GitHub release worker cache: Cloudflare R2

New shared release workers explicitly select `--r2-cache`. They require repository
secrets `R2_ACCESS_KEY_ID` and `R2_SECRET_ACCESS_KEY`; absence, embedded whitespace,
authentication failures, and malformed/missing dependency manifests fail rather
than selecting Azure or GHA. Surrounding secret whitespace is stripped and the
sanitized values are masked before use. Credentials never enter worker JSON.

Endpoint: `https://318204901782458555a243ad96f80e3f.r2.cloudflarestorage.com`

Bucket: `dl4j-cache`; signing region: `auto`.

## Cache contracts

* Compiler objects: sccache `disk,s3`, custom HTTPS endpoint, no AWS SSE header.
  Prefix remains `deeplearning4j/releases/compiler-cache/v1`.
* Toolchain archives and ccache-L0 snapshots: existing provider-neutral archive
  helper, with `release/aws/cloud-io.py` S3 transport, using multipart streaming
  transfers. Prefix remains `deeplearning4j/releases/toolchain-cache/v1`.
* Managed LLVM/MLIR dependency snapshots: authenticated R2 manifest/index/archive
  reads under `deeplearning4j/releases/dependency-cache/v2`. Migrated manifests can
  contain an Azure `publicBaseUrl`; it is discarded, never followed. Object keys,
  digests and extraction attestations are preserved.
* `snapshotIdentityBackend: azure` is historical **identity metadata**, not a
  backend choice: ccache-L0 indexes originally hashed that field. Preserving it
  avoids abandoning copied snapshot keys. Actual transports all select S3/R2.
* Patched sccache builds include `gha,azure,s3`. Only the compiled binary cache
  key changes; no compiler-object or archive namespace is invalidated. Release
  actions install only and do not start a GHA server or silently substitute an
  unpatched binary. Existing non-release GHA users retain their setup behavior.

GitHub's Maven dependency/local `.libnd4j` and executable-download caches remain
local restore accelerators, not Azure transports. Azure support and all Azure
objects remain intact. No migration or deletion is performed by worker setup.
Existing running jobs retain their old configuration; new source revisions use
R2. The separately dispatched nondeleting migration must finish before expecting
all historical cache hits.

## Remote validation only

`Release cache contracts` runs on GitHub Linux, Windows and macOS runners, without
native release fanout. Linux also runs the existing full release contracts and
an authenticated R2 check: lists/HEADs each migrated namespace, validates all
manifest index/archive references, and publishes/restores a small attested
`r2-transport-contract` archive. It adds content-addressed validation objects and
never deletes remote data. Logs are retained as workflow artifacts. These checks
do not claim native build or compiler hit-rate evidence; that requires the next
real release workers. No local tests, builds, or code generation are needed.
