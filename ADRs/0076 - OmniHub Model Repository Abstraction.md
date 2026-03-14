# OmniHub Model Repository Abstraction

## Status
**Accepted**

Proposed by: Adam Gibson (13th Mar 2026)

## Context

OmniHub was introduced in [ADR 0011](0011%20-%20OmniHub-Zoo%20Download.md) to provide a unified interface for downloading and managing pretrained models. [ADR 0014](0014%20-%20OmniHub-%20Replace%20old%20model%20zoo.md) replaced the legacy deeplearning4j-zoo with OmniHub, centralizing model downloads through a single GitHub-hosted repository (`omnihub-zoo`).

Since then, the ecosystem has expanded. OmniHub now supports HuggingFace Hub downloads (SafeTensors, GGUF) alongside the original GitHub raw-file source. However, the two sources are hardcoded directly into `OmniHubUtils` with no abstraction. Adding new model sources — such as additional HuggingFace organizations, private model registries, or alternative hosting backends — requires modifying `OmniHubUtils` each time.

This creates several problems:
- No clean way for users or downstream projects to add custom model sources
- No priority-based fallback when a model exists in multiple repositories
- HuggingFace-based models (GGUF, SafeTensors) and GitHub-hosted models (`.fb`, `.zip`) use completely different code paths with no common interface
- Model definitions in the DSL cannot specify which backend to use

## Decision

Introduce a `ModelRepository` interface, a priority-based `ModelRepositoryRegistry`, and concrete implementations for the existing GitHub and HuggingFace backends.

### ModelRepository Interface

A `ModelRepository` represents a single model source with two resolution paths:

- **Named model resolution** (`resolve`) — for models identified by name and framework (e.g. `"resnet18.fb"` in `"samediff"`)
- **HuggingFace repo resolution** (`resolveHuggingFace`) — for models identified by HF repo ID and file pattern

Each repository declares a `priority` (lower = tried first) and a `name` for identification.

### ModelRepositoryRegistry

The registry holds a priority-sorted list of `ModelRepository` instances. On resolve, it tries each repository in priority order until one succeeds. This provides automatic fallback — if a model is not found in the primary source, secondary sources are tried without caller intervention.

Default configuration:
- HuggingFace repository (priority 10) — checks a configurable HF organization for named models
- GitHub repository (priority 20) — falls back to the `omnihub-zoo` GitHub raw-file source

The default HuggingFace organization is configurable via:
- System property: `omnihub.hf.org`
- Environment variable: `OMNIHUB_HF_ORG`

### Integration with OmniHubUtils

All existing public methods on `OmniHubUtils` remain unchanged. Internally, `downloadAndLoadFromZoo` and `loadFromHuggingFace` now delegate to the registry. A new `loadFromRepository` method allows callers to bypass the fallback chain and target a specific named repository.

Users can replace the default registry via `OmniHubUtils.setRegistry()` or modify it via `ModelRepositoryRegistry.getDefault().addRepository()`.

### DSL Changes

The `Model` data class gains an optional `repository` field. When set (e.g. `"huggingface"`, `"github"`), the generated code targets that specific repository instead of using the fallback chain. `GGUFModel` and `SafeTensorsModel` DSL builders default to `repository = "huggingface"` since these formats are inherently HuggingFace-sourced. `DL4JModel` and `SameDiffModel` default to `null` (use the full fallback chain).

### New Files

- `omnihub/.../repository/ModelRepository.java` — interface
- `omnihub/.../repository/ModelRepositoryRegistry.java` — priority-based registry with singleton default
- `omnihub/.../repository/GitHubModelRepository.java` — wraps existing GitHub download logic
- `omnihub/.../repository/HuggingFaceModelRepository.java` — wraps `HuggingFaceHubDownloader`, adds org-based named model resolution

### Modified Files

- `OmnihubConfig.java` — added `OMNIHUB_HF_ORG` config, `getHuggingFaceOrg()` method
- `OmniHubUtils.java` — delegates to registry, added `setRegistry()` and `loadFromRepository()`
- `Model.kt` — added `repository: String?` field
- `ModelBuilder.kt` — GGUF/SafeTensors models default to `repository = "huggingface"`
- `ModelNamespaceGenerator.java` — generates `loadFromRepository` calls when model has explicit repository

## Consequences

### Advantages

- **Extensible**: new model sources can be added without modifying core OmniHub code
- **Fallback resilience**: if a primary source is unavailable, secondary sources are tried automatically
- **User-configurable**: organizations and enterprises can point to their own model hosting
- **Backward compatible**: all existing `OmniHubUtils` methods and `Pretrained.*` generated code continue to work unchanged
- **Clean separation**: GitHub and HuggingFace download logic is encapsulated in dedicated classes rather than mixed into a utility class

### Disadvantages

- Additional indirection layer for simple single-source downloads
- `canResolve` for HuggingFace makes a network call (HEAD request to HF API), adding latency to the fallback chain when HF is checked first but the model only exists on GitHub
- Priority values are implicit conventions (10, 20) rather than enforced ordering
