# ADR: Migrate Project Namespaces to org.eclipse.deeplearning4j using OpenRewrite

## Status

Proposed

Proposed by: Adam Gibson  (May 8, 2025)

Discussed with: Paul Dubs

## Context

The Deeplearning4j project ecosystem currently utilizes multiple root Java package namespaces, primarily `org.nd4j`, `org.deeplearning4j`, and `org.datavec`, along with existing code under `org.eclipse.deeplearning4j`, bindings to 
external libraries (like FlatBuffers, ONNX, TensorFlow, Python), and various `contrib` and `codegen` modules. This distributed namespace structure can make navigation, maintenance, and 
understanding component relationships more challenging than necessary.

The goal is to consolidate the core project codebase under a single, unified root namespace: `org.eclipse.deeplearning4j`. This aligns with the project's stewardship under the Eclipse Foundation and aims to create a clearer,
more consistent, and maintainable structure. Given the project's large scale, an automated refactoring approach using [OpenRewrite](https://docs.openrewrite.org/reference/rewrite-maven-plugin) is optimal for feasibility. This ADR proposes the strategy and 
specific rules for this migration.

This is part of a 2 phase release plan where a milestone release that has the old package names is performed followed by the major renamespacing.


## Proposal

This ADR proposes using a comprehensive [OpenRewrite recipe](https://docs.openrewrite.org/reference/rewrite-maven-plugin) to automatically refactor the project's primary Java packages into the `org.eclipse.deeplearning4j` namespace. The refactoring follows a two-phase conceptual approach:

1.  **Foundational Re-basing:** Map the main existing root packages (`org.nd4j`, `org.deeplearning4j`, `org.datavec`) to logical sub-packages under the new root (`.nd4j`, `.dl4jcore`, `.datavec`).
2.  **Architectural Refinement:** Apply more specific rules to achieve a clearer target structure, including:
    * Elevating the UI components to a top-level `org.eclipse.deeplearning4j.ui` package.
    * Consolidating ND4J backend-specific code (including former `jita` and `jcublas` into `.cuda`) under `org.eclipse.deeplearning4j.nd4j.backend.[type]`.
    * Structuring DataVec components functionally under `org.eclipse.deeplearning4j.datavec.*`.
    * Isolating runtime bindings for external libraries under `org.eclipse.deeplearning4j.bindings.*`.
    * Providing clear namespaces for `contrib` and `codegen` modules.
    * Establishing top-level `common` and `resources` packages (though full consolidation requires further steps).

The core of the proposal is the following OpenRewrite YAML recipe, which implements the automated parts of this strategy:




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# OpenRewrite Recipe for Deeplearning4j Namespace Migration                    #
# Target Root: org.eclipse.deeplearning4j                                      #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
type: org.openrewrite.Recipe
name: org.eclipse.deeplearning4j.refactor.NamespaceMigration
displayName: Migrate DL4J Project to org.eclipse.deeplearning4j Namespace
description: Comprehensive refactoring of core DL4J, ND4J, and DataVec packages under the org.eclipse.deeplearning4j root, including backend consolidation and UI elevation.

recipeList:
#--------------------------------------------------------------------------#
# Step 1: Handle Specific Component Moves (More Specific Rules First)      #
#--------------------------------------------------------------------------#

# --- UI Components -> o.e.d.ui.* ---
- org.openrewrite.java.ChangePackage:
  oldPackageName: org.deeplearning4j.ui
  newPackageName: org.eclipse.deeplearning4j.ui
  recursive: true
  # Note: This rule assumes org.deeplearning4j.ui.* should be elevated.
  # It must run before the general org.deeplearning4j -> o.e.d.dl4jcore rule.

# --- ND4J Backends -> o.e.d.nd4j.backend.[type].* ---
- org.openrewrite.java.ChangePackage:
  oldPackageName: org.nd4j.linalg.jcublas # Maps to cuda backend
  newPackageName: org.eclipse.deeplearning4j.nd4j.backend.cuda
  recursive: true
- org.openrewrite.java.ChangePackage:
  oldPackageName: org.nd4j.jita # JITA components also map to cuda backend
  newPackageName: org.eclipse.deeplearning4j.nd4j.backend.cuda # Target same new package
  recursive: true
- org.openrewrite.java.ChangePackage:
  oldPackageName: org.nd4j.linalg.cpu.nativecpu # Maps to cpu backend
  newPackageName: org.eclipse.deeplearning4j.nd4j.backend.cpu
  recursive: true
- org.openrewrite.java.ChangePackage:
  oldPackageName: org.nd4j.presets.cuda # CUDA presets
  newPackageName: org.eclipse.deeplearning4j.nd4j.backend.cuda.presets
  recursive: true
- org.openrewrite.java.ChangePackage:
  oldPackageName: org.nd4j.presets.cpu # CPU presets
  newPackageName: org.eclipse.deeplearning4j.nd4j.backend.cpu.presets
  recursive: true
- org.openrewrite.java.ChangePackage: # Minimal backend?
  oldPackageName: org.nd4j.linalg.minimal
  newPackageName: org.eclipse.deeplearning4j.nd4j.backend.minimal
  recursive: true
- org.openrewrite.java.ChangePackage:
  oldPackageName: org.nd4j.presets.minimal
  newPackageName: org.eclipse.deeplearning4j.nd4j.backend.minimal.presets
  recursive: true
- org.openrewrite.java.ChangePackage: # Common presets utils
  oldPackageName: org.nd4j.presets
  newPackageName: org.eclipse.deeplearning4j.nd4j.backend.common.presets
  recursive: true

# --- Native Ops / Bindings Facades ---
- org.openrewrite.java.ChangePackage:
  oldPackageName: org.nd4j.nativeblas # Common native facade classes like NativeOpsHolder
  newPackageName: org.eclipse.deeplearning4j.nd4j.nativeops.common
  recursive: true

# --- Runtime Bindings (Wrappers/Runners for external libs/runtimes) ---
- org.openrewrite.java.ChangePackage:
  oldPackageName: org.nd4j.onnxruntime # ONNX Runtime Runner
  newPackageName: org.eclipse.deeplearning4j.bindings.onnx.runtime
  recursive: true
- org.openrewrite.java.ChangePackage:
  oldPackageName: org.nd4j.tensorflow.conversion # TF Runtime Runner/Converter
  newPackageName: org.eclipse.deeplearning4j.bindings.tensorflow.runtime
  recursive: true
- org.openrewrite.java.ChangePackage:
  oldPackageName: org.nd4j.tensorflowlite # TF Lite Runner
  newPackageName: org.eclipse.deeplearning4j.bindings.tensorflowlite.runtime
  recursive: true
- org.openrewrite.java.ChangePackage:
  oldPackageName: org.nd4j.tvm # TVM Runner
  newPackageName: org.eclipse.deeplearning4j.bindings.tvm.runtime
  recursive: true
- org.openrewrite.java.ChangePackage:
  oldPackageName: org.nd4j.python4j # Python Bindings
  newPackageName: org.eclipse.deeplearning4j.bindings.python.api # Map core to .api
  recursive: true # Handles subpackages like .numpy unless more specific rules added

# --- Specific Contrib Modules (Using explicit contrib namespacing) ---
# Note: fileMatcher might be needed in practice to limit these rules to specific module paths
- org.openrewrite.java.ChangePackage:
  oldPackageName: org.nd4j # For classes directly in org.nd4j within nd4j-benchmark
  newPackageName: org.eclipse.deeplearning4j.contrib.benchmark.nd4j
  recursive: false # Avoid affecting subpackages handled below
- org.openrewrite.java.ChangePackage:
  oldPackageName: org.nd4j.bypass # Sub-package in nd4j-benchmark
  newPackageName: org.eclipse.deeplearning4j.contrib.benchmark.nd4j.bypass
  recursive: true
- org.openrewrite.java.ChangePackage:
  oldPackageName: org.nd4j.memorypressure # Sub-package in nd4j-benchmark
  newPackageName: org.eclipse.deeplearning4j.contrib.benchmark.nd4j.memorypressure
  recursive: true
- org.openrewrite.java.ChangePackage:
  oldPackageName: org.nd4j # For classes directly in org.nd4j within version-updater
  newPackageName: org.eclipse.deeplearning4j.contrib.versionupdater
  recursive: false
- org.openrewrite.java.ChangePackage:
  oldPackageName: org.nd4j.fileupdater # Sub-package in version-updater
  newPackageName: org.eclipse.deeplearning4j.contrib.versionupdater.fileupdater
  recursive: true
- org.openrewrite.java.ChangePackage: # For contrib blas-lapack-gen
  oldPackageName: org.deeplearning4j
  newPackageName: org.eclipse.deeplearning4j.contrib.blaslapackgen
  recursive: true
- org.openrewrite.java.ChangePackage: # For contrib nd4j-log-analyzer
  oldPackageName: org.nd4j.interceptor
  newPackageName: org.eclipse.deeplearning4j.contrib.nd4jloganalyzer.interceptor
  recursive: true

# --- Specific Codegen Modules ---
# Note: fileMatcher might be needed in practice
- org.openrewrite.java.ChangePackage: # For libnd4j-gen
  oldPackageName: org.nd4j.descriptor
  newPackageName: org.eclipse.deeplearning4j.codegen.descriptor
  recursive: true
- org.openrewrite.java.ChangePackage: # For op-codegen
  oldPackageName: org.nd4j.codegen
  newPackageName: org.eclipse.deeplearning4j.codegen.op
  recursive: true
- org.openrewrite.java.ChangePackage: # For codegen blas-lapack-generator
  oldPackageName: org.deeplearning4j
  newPackageName: org.eclipse.deeplearning4j.codegen.blaslapack
  recursive: true

#--------------------------------------------------------------------------#
# Step 2: Apply General Re-basing Rules (Less Specific Rules Last)       #
#--------------------------------------------------------------------------#

- org.openrewrite.java.ChangePackage:
  oldPackageName: org.nd4j # General ND4J code not caught by specific rules above
  newPackageName: org.eclipse.deeplearning4j.nd4j
  recursive: true
  # This will map remaining org.nd4j.* like linalg.api.*, samediff.*, common.*, etc.

- org.openrewrite.java.ChangePackage:
  oldPackageName: org.deeplearning4j # General DL4J Core code (excluding UI already handled)
  newPackageName: org.eclipse.deeplearning4j.dl4jcore
  recursive: true
  # This will map nn.*, models.*, datasets.*, eval.*, nlp.*, etc.

- org.openrewrite.java.ChangePackage:
  oldPackageName: org.datavec # General DataVec code
  newPackageName: org.eclipse.deeplearning4j.datavec
  recursive: true
  # This will map api.*, transform.*, image.*, arrow.*, etc.

#--------------------------------------------------------------------------#
# Step 3: Exclusions / Manual Handling (Commented Out for Safety)          #
#--------------------------------------------------------------------------#

# --- DO NOT AUTOMATICALLY RUN THESE for generated binding specs ---
# --- Prefer changing generator configurations ---
# - org.openrewrite.java.ChangePackage:
#     oldPackageName: com.google.flatbuffers
#     newPackageName: org.eclipse.deeplearning4j.bindings.flatbuffers.spec # Or .core
#     recursive: true
# - org.openrewrite.java.ChangePackage:
#     oldPackageName: onnx
#     newPackageName: org.eclipse.deeplearning4j.bindings.onnx.spec
#     recursive: true
# - org.openrewrite.java.ChangePackage:
#     oldPackageName: tensorflow # Includes tensorflow.framework, tensorflow.eager etc
#     newPackageName: org.eclipse.deeplearning4j.bindings.tensorflow.spec
#     recursive: true

#--------------------------------------------------------------------------#
# Step 4: Potential Cleanup (Run Separately After Migration)               #
#--------------------------------------------------------------------------#

# - org.openrewrite.java.cleanup.OrganizeImports
# - org.openrewrite.java.cleanup.RemoveUnusedImports
# - org.openrewrite.java.format.AutoFormat

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# End of Recipe                                                                #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


## Consequences

### Advantages

* **Namespace Consistency:** Provides a single, unified root namespace (`org.eclipse.deeplearning4j`) for the entire project, aligning with Eclipse Foundation stewardship.
* **Improved Structure:** The refined Phase 2 structure (explicit backends, UI, common components, bindings) enhances architectural clarity and modularity compared to the original scattered namespaces.
* **Easier Navigation & Maintenance:** A consistent structure makes it easier for developers to find code, understand component relationships, and perform maintenance.
* **Foundation for Future Work:** Creates a cleaner base for developing new features and potentially consolidating shared functionality (e.g., in common utilities, resource management).

### Disadvantages

* **Significant Initial Effort:** Running this large-scale refactoring requires careful setup, dry runs, review, and validation.
* **Potential for Breakage:** Automated refactoring on this scale carries inherent risks. Build script issues, reflection usage, non-Java file references (like in XML or properties), and complex type resolution might lead to compilation errors or runtime issues requiring manual fixes.
* **Testing Burden:** Requires executing the full test suite (unit, integration, platform-specific) to ensure correctness after refactoring. Test code itself will also be refactored and needs verification.
* **External Bindings Complexity:** Handling generated code or bindings for libraries like FlatBuffers, ONNX, and TensorFlow requires careful consideration beyond simple package renaming; modifying generator configurations is preferred.
* **Learning Curve:** Teams unfamiliar with OpenRewrite will need to learn its usage and recipe development, especially for any necessary follow-up complex refactorings.
* **Merge Conflicts:** This constitutes a large change, potentially creating significant merge conflicts for ongoing feature branches. Coordination is essential.

### Technical details about the Namespace Refactoring

* **Tooling:** The proposal relies on OpenRewrite and its `org.openrewrite.java.ChangePackage` recipe.
* **Recipe Structure:** The YAML recipe uses a `recipeList` and applies more specific package mapping rules *before* more general ones to handle structural changes like backend separation and UI elevation correctly.
* **`ChangePackage` Limitations:** This recipe primarily uses `ChangePackage`. Achieving the full Phase 2 vision, especially consolidating classes from multiple old locations into a single new package (e.g., common utilities), may require additional recipes using `MoveClass`, `MovePackage`, or custom Java-based recipes after this initial migration.
* **Exclusions:** Generated binding specifications (`com.google.flatbuffers`, `onnx.*`, `tensorflow.*`) are intentionally excluded from automated changes due to high risk. Runtime wrappers/runners for these *are* included in the refactoring.
* **Process:** The recommended process involves: backup -> dry run -> review -> iterative recipe refinement (if needed) -> apply changes -> compile -> test -> manual code review -> commit.

## Discussion

The proposed structure aims for a balance between respecting the original component boundaries (ND4J, DL4J Core, DataVec) and creating a more modern, cohesive structure under the Eclipse Foundation namespace. Key decisions reflected in the proposal:

* Explicitly separating backend implementations (`.nd4j.backend.[type]`).
* Elevating UI to a top-level component (`.ui`).
* Designating clear homes for bindings (`.bindings`), contrib (`.contrib`), and codegen (`.codegen`).
* Establishing placeholders for consolidated common code (`.common`) and resource management (`.resources`), acknowledging full consolidation is a subsequent step.
* Using `.dl4jcore` to house the bulk of the original `org.deeplearning4j` code to distinguish it clearly within the new structure.

Further discussion within the development team is needed to confirm these target structures and plan the implementation, particularly the steps required beyond the initial automated recipe application for deeper Phase 2 refinements.