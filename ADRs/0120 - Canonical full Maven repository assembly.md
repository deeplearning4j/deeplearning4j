# ADR 0120: Canonical full Maven repository assembly

## Status

Implemented; full GitHub Actions qualification and release approval pending.

## Context

The snapshot dispatcher builds selected native matrices. Its default four CPU base variants are not a complete Maven release, and a union of worker repositories does not supply the default Java reactor. Native workers also produce different unclassified POMs at shared coordinates. In particular, CUDA 12.6, 12.9, and 13.1 previously wrote incompatible dependencies into one common-module GAV.

The release namespace is `org.eclipse.deeplearning4j`. A release must not silently choose conflicting metadata, use cached snapshots to conceal missing components, or publish automatically. This decision concerns Maven assembly, not Java package migration or standalone SDK distribution.

## Decision

- Add an explicit `workflow=all` selection derived from the canonical provider plan, currently 66 variants. Reject partial selection. Run release contract tests before native fanout.
- Pin source to a full commit SHA. Require the exact worker set, matching source/version/run identity and plan digests, completion receipts, and archive attestations. Failed-job retries can reuse successful artifacts only from earlier attempts of the same run.
- Give each native classifier a unique owner and each shared native component a declared owner. Separate CUDA common-module coordinates by CUDA version. Keep platform dependency declarations aligned with the published classifier inventory.
- Assemble in a fresh, uncached Maven local repository. Seed attested native outputs and their actual parent chains, then replace bootstrap parents by building the default Java reactor from the same source. Build platform aggregators separately without native profiles or `--also-make`.
- Require main artifacts, parent closure, POM metadata, and generated sources/javadoc for non-SNAPSHOT components. Record actual effective Maven reactor models rather than counting inactive modules found on disk.
- Resolve each shipped binary in its own consumer module to avoid dependency mediation between components concealing missing dependencies. Reject newly fetched DL4J artifacts outside this run's inventory.
- Produce a hash/size/ownership manifest and retain the build-only Maven repository for inspection. Do not include standalone SDK archives in full Maven worker uploads.
- Keep metadata generation separate from signing. `dryRun=true` neither signs nor uploads. Actual release staging uses Central `USER_MANAGED`; the Maven plugin defaults `autoPublish` to false. No workflow requests automatic release.
- Generate release Javadoc from the normal project sources without a release-only delombok pipeline. Declare existing public Lombok builder types in source where Javadoc needs to resolve them, leaving member generation to Lombok during compilation. Repair documentation syntax and references at source, and preserve strict Javadoc validation without package exclusions or error suppression.

## Consequences and qualification

The assembly has an explicit coverage contract: the canonical native matrix, default Java reactor, their parents, and the selected platform aggregators. Optional Spark/profile-only modules and optional AOT/standalone SDK outputs are not implicitly certified.

Consumer resolution is evaluated on the Linux assembly host. It does not certify every OS-activated Maven profile, optional dependency, or native runtime behavior. Contract tests do not substitute for a completed matrix build.

Selected-worker publication recovery is not a substitute for full assembly. Failed full assembly jobs should be rerun in the original Actions run while successful worker artifacts remain retained.

A release still requires successful matrix and consumer evidence, signing/key validation, namespace authorization, and a signed-bundle size check. The existing uploader rejects bundles at or above Central's 1 GB limit; larger releases need qualified component-preserving partitioning. Dry-run success is not authorization to stage or publish. An explicitly approved staging operation must reach Central `VALIDATED`, followed by human publication approval.

## References

- `release/github/full-repository.py`
- `.github/workflows/_release-worker.yml`
- `release/github/README.md`
- `platform-tests/test_full_release_repository.py`
- `platform-tests/test_release_publication_safety.py`
