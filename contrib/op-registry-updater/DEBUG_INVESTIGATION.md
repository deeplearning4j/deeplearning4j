# EmbedLayerNormalization PreImportHook Investigation

## Problem Statement

The `EmbedLayerNormalization.kt` implementation with `@PreHookRule` annotation is not being processed during op registry updates, despite being properly annotated and having a corresponding `noop` mapping in the ONNX op declarations.

## Analysis

### Current Setup

1. **File Location**: `/nd4j/samediff-import/samediff-import-onnx/src/main/kotlin/org/nd4j/samediff/frameworkimport/onnx/definitions/implementations/EmbedLayerNormalization.kt`

2. **Annotation**: 
   ```kotlin
   @PreHookRule(nodeNames = [],opNames = ["EmbedLayerNormalization"],frameworkName = "onnx")
   ```

3. **Registry Mapping**: 
   ```kotlin
   val embedLayerNormalization = OnnxMappingProcess(
           opName = "noop",
           opMappingRegistry = onnxOpRegistry,
           inputFrameworkOpName = "EmbedLayerNormalization"
   )
   ```

### How PreImportHook Discovery Works

1. **ClassGraph Scanning**: The `ImportReflectionCache` uses ClassGraph to scan the entire classpath for classes implementing `PreImportHook` with `@PreHookRule` annotations.

2. **Cache Population**: Found hooks are stored in `preProcessRuleImplementationsByOp` table indexed by framework name and op name.

3. **Registry Lookup**: When `OpMappingRegistry.lookupOpMappingProcess()` is called, it first checks for registered mapping processes, then falls back to checking the cache for PreImportHooks.

### Potential Root Causes

#### 1. Classpath Issues
The implementations directory may not be in the classpath when the registry updater runs:
- The standalone module may not be including the compiled implementations
- Build order issues preventing proper compilation
- Missing dependency on samediff-import-onnx at runtime

#### 2. ClassGraph Scan Scope
The ClassGraph scanner may not be reaching the implementations package:
- Scan may be limited to specific packages
- Module boundaries preventing cross-module scanning
- JAR packaging issues

#### 3. Annotation Processing
The `@PreHookRule` annotation may not be processed correctly:
- Annotation retention issues
- Kotlin/Java interop problems
- Build-time vs runtime annotation availability

#### 4. Cache Initialization Timing
The ImportReflectionCache may not be properly initialized:
- System property `INIT_IMPORT_REFLECTION_CACHE` may be false
- Cache loading may fail silently
- Timing issues with when cache is populated vs when it's accessed

## Debug Strategy

### Phase 1: Verify Basic Setup

1. **Run with Debug Mode**:
   ```bash
   cd contrib/op-registry-updater
   ./update-op-registry.sh --framework onnx --debug --validate-only
   ```

2. **Check Expected Outputs**:
   - Does ClassGraph find any PreImportHooks for ONNX?
   - Is EmbedLayerNormalization specifically found?
   - Are there any errors during cache initialization?

### Phase 2: Classpath Investigation

1. **Verify Class Availability**:
   ```bash
   # Check if class can be loaded manually
   java -cp "target/classes:$(mvn dependency:build-classpath -q -Dmdep.outputFile=/dev/stdout)" \
     -c "Class.forName('org.nd4j.samediff.frameworkimport.onnx.definitions.implementations.EmbedLayerNormalization')"
   ```

2. **Check JAR Contents**:
   ```bash
   # If using shaded JAR, check contents
   jar -tf target/op-registry-updater-1.0.0-SNAPSHOT.jar | grep EmbedLayerNormalization
   ```

### Phase 3: Dependency Analysis

1. **Verify samediff-import-onnx Inclusion**:
   - Check that the pom.xml includes samediff-import-onnx as a dependency
   - Ensure the dependency includes the implementations package
   - Verify transitive dependencies are resolved

2. **Maven Dependency Tree**:
   ```bash
   mvn dependency:tree | grep samediff-import-onnx
   ```

### Phase 4: Manual Hook Registration

If automatic discovery fails, consider manual registration:

```kotlin
// In OpRegistryUpdater, add manual hook registration
private fun ensurePreImportHooksLoaded() {
    try {
        val embedLayerNormClass = Class.forName("org.nd4j.samediff.frameworkimport.onnx.definitions.implementations.EmbedLayerNormalization")
        val instance = embedLayerNormClass.getDeclaredConstructor().newInstance() as PreImportHook
        
        val hooks = ImportReflectionCache.preProcessRuleImplementationsByOp.get("onnx", "EmbedLayerNormalization")
        if (hooks == null || hooks.isEmpty()) {
            println("  🔧 Manually registering EmbedLayerNormalization PreImportHook")
            ImportReflectionCache.preProcessRuleImplementationsByOp.put("onnx", "EmbedLayerNormalization", mutableListOf(instance))
        }
    } catch (e: Exception) {
        println("  ❌ Failed to manually register EmbedLayerNormalization: ${e.message}")
    }
}
```

## Enhanced Debug Version

The updated standalone tool now includes:

1. **Debug Flag**: `--debug` option to enable detailed PreImportHook discovery logging
2. **Hook Discovery Check**: Explicit check for EmbedLayerNormalization hooks
3. **Manual Class Loading**: Fallback attempt to manually load the class and check annotations
4. **Detailed Reporting**: Shows noop mappings vs actual PreImportHooks found

## Recommended Actions

1. **Immediate Testing**:
   ```bash
   cd contrib/op-registry-updater
   chmod +x update-op-registry.sh
   ./update-op-registry.sh --framework onnx --debug --validate-only
   ```

2. **Check Build Dependencies**:
   Ensure the standalone module properly builds and includes all required classes:
   ```bash
   mvn clean compile -X  # Verbose output
   ```

3. **Verify Registry Behavior**:
   The debug output should show whether:
   - EmbedLayerNormalization is found as a "noop" mapping
   - Corresponding PreImportHooks are discovered
   - Any ClassGraph scanning issues

4. **Fix Based on Findings**:
   - If classpath issue: Update dependencies or build order
   - If scanning issue: Adjust ClassGraph configuration
   - If timing issue: Force cache reload at appropriate time

## Expected Fix

Most likely the issue is that the samediff-import-onnx module needs to be fully compiled and available in the classpath when the standalone registry updater runs. The fix may involve:

1. Ensuring proper build order in the shell script
2. Adding explicit dependency management in the standalone pom.xml
3. Verifying that the implementations package is included in the compiled artifacts

The debug output will reveal exactly where the breakdown occurs in the PreImportHook discovery process.
