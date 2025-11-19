# OP Registry Migration Guide

This document outlines the migration from the platform-tests based OP registry update system to the new standalone module.

## Summary of Changes

### Files Created
1. **`contrib/op-registry-updater/`** - New standalone Maven module
   - `pom.xml` - Maven configuration with minimal dependencies
   - `src/main/kotlin/org/eclipse/deeplearning4j/contrib/OpRegistryUpdater.kt` - Main application class
   - `update-op-registry.sh` - Standalone shell script
   - `README.md` - Comprehensive documentation

### Files That Can Be Deprecated/Removed
1. **`platform-tests/src/main/kotlin/org/eclipse/deeplearning4j/frameworkimport/runner/OpRegistryUpdater.kt`**
   - Replaced by the standalone version in contrib/
2. **`update-op-registry.sh` (project root)**
   - Replaced by the standalone version in contrib/op-registry-updater/
3. **`OP_REGISTRY_UPDATE_README.md` (project root)**
   - Information consolidated into the new module's README

## Migration Instructions

### For Users

#### Before (Old Method):
```bash
# From project root
./update-op-registry.sh --framework tensorflow
```

#### After (New Method):
```bash
# Navigate to standalone module
cd contrib/op-registry-updater

# Make script executable (first time only)
chmod +x update-op-registry.sh

# Run the updater
./update-op-registry.sh --framework tensorflow
```

### Key Benefits

1. **No Platform-Tests Dependency**: 
   - Old system required building the entire platform-tests module
   - New system only builds required framework import dependencies

2. **Faster Build Times**:
   - Old: ~5-10 minutes (full platform-tests build)
   - New: ~1-2 minutes (minimal dependencies)

3. **Cleaner Separation**:
   - OP registry updates are now separate from test infrastructure
   - Purpose-built tool with clear CLI interface

4. **Better Maintainability**:
   - Standalone module is easier to understand and maintain
   - No confusion with test-related code

## Usage Examples

### Quick Start
```bash
cd contrib/op-registry-updater
./update-op-registry.sh
```

### Framework-Specific Updates
```bash
# TensorFlow only
./update-op-registry.sh --framework tensorflow

# ONNX only  
./update-op-registry.sh --framework onnx
```

### Validation Only
```bash
# Validate without saving (useful for CI/PR checks)
./update-op-registry.sh --validate-only
```

### Using Maven Directly
```bash
cd contrib/op-registry-updater

# Compile and run
mvn compile exec:java

# With arguments
mvn compile exec:java -Dexec.args="--framework=tensorflow --validate-only"
```

### Using Executable JAR
```bash
cd contrib/op-registry-updater

# Build JAR
mvn clean package

# Run JAR
java -jar target/op-registry-updater-1.0.0-SNAPSHOT.jar --framework=all
```

## Technical Details

### Dependencies (Minimal)
The new module only depends on:
- `samediff-import-api`
- `samediff-import-tensorflow`
- `samediff-import-onnx`
- `kotlin-stdlib-jdk8`
- `slf4j-simple`

### Output Files
Both old and new systems generate the same output files:
- TensorFlow: `nd4j/samediff-import/samediff-import-tensorflow/src/main/resources/`
- ONNX: `nd4j/samediff-import/samediff-import-onnx/src/main/resources/`

### Functionality
The new system provides the same functionality as the old:
- Registry validation
- Configuration file generation
- Error reporting
- Framework-specific processing

## CI/CD Integration

### Before
```bash
./update-op-registry.sh --validate-only  # From project root
```

### After
```bash
cd contrib/op-registry-updater
./update-op-registry.sh --validate-only
```

## Cleanup Recommendations

Once the new system is verified to work correctly, consider:

1. **Remove old files**:
   - `update-op-registry.sh` (project root)
   - `OP_REGISTRY_UPDATE_README.md` (project root)
   - `platform-tests/src/main/kotlin/org/eclipse/deeplearning4j/frameworkimport/runner/OpRegistryUpdater.kt`

2. **Update documentation**:
   - Update any references to the old system
   - Point users to the new contrib module

3. **Update CI/CD pipelines**:
   - Change paths in build scripts
   - Update automation that uses the registry updater

## Testing the Migration

### Validation Steps
1. Build the new module:
   ```bash
   cd contrib/op-registry-updater
   mvn clean compile
   ```

2. Test validation mode:
   ```bash
   ./update-op-registry.sh --validate-only
   ```

3. Compare outputs with old system:
   ```bash
   # Run old system (validation only)
   cd <project-root>
   ./update-op-registry.sh --validate-only
   
   # Run new system (validation only)
   cd contrib/op-registry-updater
   ./update-op-registry.sh --validate-only
   ```

4. Test actual updates in a separate branch:
   ```bash
   cd contrib/op-registry-updater
   ./update-op-registry.sh
   # Check git diff for expected file changes
   ```

## Support

If you encounter issues with the migration:

1. Check the [README](contrib/op-registry-updater/README.md) for detailed usage instructions
2. Ensure all prerequisites are met (Java 11+, Maven 3.6+)
3. Try the `--verbose` flag for detailed output
4. Use `--clean` flag if builds fail

The new standalone system should be a drop-in replacement with improved performance and maintainability.
