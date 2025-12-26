# ADR: Namespace Migration from org.deeplearning4j to org.eclipse.deeplearning4j

## Status

Proposed

Proposed by: Adam Gibson (January 2025)

Discussed with: Development Team

## Context

The Deeplearning4j project faces a critical infrastructure change with the impending shutdown of OSSRH (OSS Repository Hosting) that currently hosts our Maven artifacts under the `org.deeplearning4j`, `org.nd4j`, and `org.datavec` namespaces. As an Eclipse Foundation project since 2023, we need to migrate to the standardized `org.eclipse.deeplearning4j` namespace to align with Eclipse Foundation standards and ensure continued artifact distribution through Maven Central's new publishing infrastructure.

This migration affects:
- All Java packages across the monorepo (nd4j, datavec, deeplearning4j modules)
- Maven groupIds in all POM files
- Import statements in thousands of Java/Kotlin source files
- User code that depends on our libraries
- Build scripts and CI/CD pipelines
- Documentation and examples

The challenge is compounded by:
1. The need to maintain backward compatibility during transition
2. The scale of the codebase (864+ files across multiple modules)
3. Active development that continues during migration
4. External dependencies and bindings (ONNX, TensorFlow, FlatBuffers)

## Proposal

This ADR proposes a two-phase release strategy to migrate from current namespaces to `org.eclipse.deeplearning4j`:

### Phase 1: Milestone Release with Current Namespaces
- Create a final milestone release (1.0.0-M3) under existing namespaces
- Update only Maven groupIds to `org.eclipse.deeplearning4j`
- Keep Java package names unchanged
- This provides users a stable version before the breaking change

### Phase 2: Major Release with Full Migration
- Perform comprehensive package refactoring using OpenRewrite
- Update all Java/Kotlin packages to new namespace
- Release as 1.0.0 under Eclipse namespace
- Provide migration guide and automated tooling for users

### Implementation Strategy

The migration uses OpenRewrite for automated refactoring with specific rules:

1. **Primary Namespace Mappings**:
   - `org.nd4j.*` → `org.eclipse.deeplearning4j.nd4j.*`
   - `org.deeplearning4j.*` → `org.eclipse.deeplearning4j.dl4j.*`
   - `org.datavec.*` → `org.eclipse.deeplearning4j.datavec.*`

2. **Component-Specific Mappings**:
   - UI components elevated: `org.deeplearning4j.ui.*` → `org.eclipse.deeplearning4j.ui.*`
   - Backend consolidation: `org.nd4j.linalg.jcublas.*` → `org.eclipse.deeplearning4j.nd4j.backend.cuda.*`
   - Native operations: `org.nd4j.nativeblas.*` → `org.eclipse.deeplearning4j.nd4j.nativeops.*`
   - Runtime bindings: `org.nd4j.onnxruntime.*` → `org.eclipse.deeplearning4j.bindings.onnx.runtime.*`

3. **Special Handling**:
   - Generated code (protobuf, flatbuffers) remains unchanged initially
   - External bindings updated through generator configuration changes
   - Shaded dependencies (jackson, protobuf) migrate with main code

### Maven POM Changes

All POMs update their groupId:
```xml
<!-- Before -->
<groupId>org.nd4j</groupId>
<artifactId>nd4j-api</artifactId>

<!-- After Phase 1 -->
<groupId>org.eclipse.deeplearning4j</groupId>
<artifactId>nd4j-api</artifactId>

<!-- Package imports remain unchanged in Phase 1 -->
```

### Module Structure Preservation

Despite namespace changes, the module structure remains:
- `/nd4j` - Linear algebra and autodiff
- `/datavec` - Data preprocessing  
- `/deeplearning4j` - Neural network implementations
- `/libnd4j` - Native C++ operations
- `/python4j` - Python interop
- `/omnihub` - Model hub integration

## Consequences

### Advantages

1. **Compliance**: Aligns with Eclipse Foundation standards and ensures continued Maven Central access
2. **Consistency**: Single root namespace improves project cohesion and discoverability
3. **Future-Proof**: Positions project for long-term sustainability under Eclipse
4. **Automated Migration**: OpenRewrite minimizes manual effort and errors
5. **Clear Organization**: Better separation of core components, backends, and bindings

### Disadvantages  

1. **Breaking Change**: All user code requires import statement updates
2. **Ecosystem Impact**: All downstream projects, examples, and documentation need updates
3. **Migration Burden**: Users must plan and execute migration for their codebases
4. **Temporary Disruption**: Period of confusion during transition between releases
5. **Build Complexity**: Two-phase release adds complexity to release process

### Technical Details

**Release Timeline**:
- Milestone Release (1.0.0-M3): Immediate - groupId change only
- Major Release (1.0.0): Following milestone by 3-6 months - full package migration

**User Migration Support**:
- Automated migration tool using OpenRewrite recipes
- Detailed migration guide with common patterns
- Compatibility matrix for mixed dependency scenarios
- Support period for critical fixes to milestone release

**Build Infrastructure Changes**:
- Update CI/CD pipelines for new artifact coordinates
- Configure Maven Central publishing under Eclipse namespace
- Update artifact signing with Eclipse certificates
- Maintain parallel documentation for both versions temporarily

## Discussion

The two-phase approach balances the urgent need to address OSSRH shutdown with the responsibility to provide users adequate transition time. The milestone release serves as a stable checkpoint, while the major release completes the migration.

Key considerations discussed:

1. **Why not maintain compatibility packages?** 
   - Dual maintenance burden unsustainable
   - Confusion with two parallel namespaces
   - Clean break clearer for users

2. **Why not delay until 2.0?**
   - OSSRH shutdown timeline forces action
   - 1.0 already represents major milestone
   - Further delay increases technical debt

3. **Why preserve module names (nd4j, datavec)?**
   - Maintains conceptual continuity
   - Reduces documentation updates
   - Familiar to existing users

The migration represents a one-time disruption that positions Deeplearning4j for sustainable development under Eclipse Foundation governance. The automated tooling and phased approach minimize impact while ensuring compliance with infrastructure requirements.