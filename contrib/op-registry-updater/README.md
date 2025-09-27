# OP Registry Updater - Standalone Tool

This standalone Maven module provides a streamlined way to update framework import operation registry configurations without depending on the platform-tests module.

## Overview

The framework import system in DeepLearning4J supports importing models from TensorFlow and ONNX. Each framework has a registry of available operations that can be imported, along with their mapping rules from the source framework to ND4J operations.

Previously, updating these registries required:
1. Building the entire platform-tests module 
2. Running specific test methods:
   - `TestTensorflowProcessLoader.saveTest()` for TensorFlow
   - `TestOnnxProcessLoader.saveTest()` for ONNX

This standalone tool eliminates those dependencies and provides a clean, purpose-built solution.

## Features

- **Standalone**: No dependency on platform-tests module
- **Lightweight**: Only builds required framework import dependencies
- **Flexible**: Supports both Maven execution and standalone JAR
- **Validation**: Can validate existing configurations without saving
- **Framework-specific**: Update TensorFlow, ONNX, or both
- **Debug Mode**: Troubleshoot PreImportHook discovery issues

## Quick Start

### Using the Shell Script (Recommended)

```bash
# Navigate to the module directory
cd contrib/op-registry-updater

# Make the script executable (Unix/Linux/macOS)
chmod +x update-op-registry.sh

# Update both TensorFlow and ONNX registries
./update-op-registry.sh

# Update only TensorFlow registry
./update-op-registry.sh --framework tensorflow

# Validate without saving changes
./update-op-registry.sh --validate-only
```

### Using Maven Directly

```bash
cd contrib/op-registry-updater

# Compile and run
mvn compile exec:java

# Run with specific framework
mvn compile exec:java -Dexec.args="--framework=tensorflow"

# Validate only
mvn compile exec:java -Dexec.args="--validate-only"
```

### Using Executable JAR

```bash
cd contrib/op-registry-updater

# Build the executable JAR
mvn clean package

# Run the JAR
java -jar target/op-registry-updater-1.0.0-SNAPSHOT.jar --framework=all
```

## Usage Options

### Shell Script Options

- `--framework <n>`: Update specific framework (tensorflow|onnx|all). Default: all
- `--validate-only`: Only validate existing configs without saving. Default: false
- `--clean`: Perform clean build before running. Default: false
- `--verbose`: Enable verbose output. Default: false
- `--debug`: Enable debug output for PreImportHook discovery. Default: false
- `--use-jar`: Use executable JAR instead of maven exec:java. Default: false
- `--help, -h`: Show help message

### Application Options

- `--framework=<n>`: Update specific framework (tensorflow|onnx|all)
- `--validate-only`: Only validate without saving
- `--debug`: Enable debug output for PreImportHook discovery
- `--help`: Show help message

## Examples

### Complete Examples

```bash
# Update all registries with clean build
./update-op-registry.sh --clean

# Update TensorFlow only with verbose output
./update-op-registry.sh --framework tensorflow --verbose

# Debug ONNX PreImportHook discovery
./update-op-registry.sh --framework onnx --debug --validate-only

# Validate ONNX registry using executable JAR
./update-op-registry.sh --framework onnx --validate-only --use-jar

# Quick validation of all registries
./update-op-registry.sh --validate-only

# Maven execution with custom arguments
mvn compile exec:java -Dexec.args="--framework=onnx --validate-only"
```

## Troubleshooting PreImportHook Issues

### Debug Mode

If you're experiencing issues with PreImportHook implementations (like `EmbedLayerNormalization`) not being recognized:

```bash
# Run with debug mode to see hook discovery details
./update-op-registry.sh --framework onnx --debug --validate-only
```

The debug output will show:
- Total PreImportHook entries found in cache
- Specific hooks found for each framework
- Whether classes can be loaded manually
- Annotation discovery status

### Common Issues

**"No PreImportHooks found for [OpName]"**
- The implementation class may not be in the classpath
- Annotations might not be processed correctly
- Build order issues with dependencies

**"EmbedLayerNormalization class not found"**
- samediff-import-onnx module needs to be built first
- Check that the implementations package is included
- Verify dependency resolution

**Debug Output Shows Missing Hooks**
- Check that all dependencies are properly compiled
- Ensure ClassGraph scanning includes all necessary packages
- Verify annotation retention and processing

### Manual Investigation

See `DEBUG_INVESTIGATION.md` for detailed troubleshooting steps and potential fixes.

## Requirements

- **Java**: 11 or higher
- **Maven**: 3.6 or higher
- **Memory**: Recommend 8GB+ heap for large builds
- **Permissions**: Write access to framework import resource directories

## What It Does

### For TensorFlow:
1. Loads the TensorFlow op registry from `org.nd4j.samediff.frameworkimport.tensorflow.definitions.registry()`
2. Validates each registered operation mapping by:
   - Serializing the existing mapping process
   - Creating a new process from the serialized data
   - Comparing the original and recreated processes
3. Reports validation results with clear indicators (✓, ⚠, ℹ)
4. Saves the registry using `registry().saveProcessesAndRuleSet()`

### For ONNX:
1. Loads the ONNX op registry from `org.nd4j.samediff.frameworkimport.onnx.definitions.registry()`
2. Performs the same validation process as TensorFlow
3. Identifies `noop` mappings that rely on PreImportHooks
4. Reports on PreImportHook discovery and availability
5. Saves the registry and verifies it can be reloaded from file

### Output Files
The tool generates configuration files in the appropriate framework import modules:
- **TensorFlow**: `nd4j/samediff-import/samediff-import-tensorflow/src/main/resources/`
- **ONNX**: `nd4j/samediff-import/samediff-import-onnx/src/main/resources/`

Generated files are typically large protocol buffer text files and are not meant to be manually edited.

## Module Structure

```
contrib/op-registry-updater/
├── pom.xml                                    # Maven configuration
├── update-op-registry.sh                     # Standalone shell script
├── README.md                                 # This file
├── MIGRATION.md                              # Migration guide from old system
├── DEBUG_INVESTIGATION.md                    # Detailed troubleshooting guide
└── src/main/kotlin/
    └── org/eclipse/deeplearning4j/contrib/
        └── OpRegistryUpdater.kt               # Main application class
```

## Dependencies

This module only depends on:
- `samediff-import-api`: Core framework import API
- `samediff-import-tensorflow`: TensorFlow import functionality  
- `samediff-import-onnx`: ONNX import functionality
- `kotlin-stdlib-jdk8`: Kotlin runtime
- `slf4j-simple`: Logging

No dependency on platform-tests or other testing modules.

## Migration from Previous System

### Before (platform-tests dependency):
```bash
# Old method - required platform-tests module
./update-op-registry.sh  # From project root
```

### After (standalone):
```bash
# New method - standalone module
cd contrib/op-registry-updater
./update-op-registry.sh
```

### Benefits of Migration:
- **Faster builds**: Only builds required dependencies
- **Cleaner separation**: Registry updates independent of tests
- **Better maintainability**: Purpose-built tool with clear interface
- **Reduced complexity**: No need to understand platform-tests structure
- **Debug capabilities**: Built-in troubleshooting for PreImportHook issues

## Integration

### CI/CD Integration
```bash
# Validate all registries without saving (for PR checks)
cd contrib/op-registry-updater
./update-op-registry.sh --validate-only

# Update registries in release builds
./update-op-registry.sh --framework all
```

### Development Workflow
1. Add new op mappings to the appropriate registry files
2. Run validation: `./update-op-registry.sh --validate-only`
3. Debug any issues: `./update-op-registry.sh --debug --validate-only`
4. Fix any validation errors
5. Update the configs: `./update-op-registry.sh`
6. Commit the updated configuration files

## Troubleshooting

### Common Issues

**"Module not found"**
- Ensure you're running from `contrib/op-registry-updater/`
- Check that the module's `pom.xml` exists

**"Build failed for dependency"**
- Try with `--clean` flag
- Check that Java and Maven versions meet requirements
- Ensure sufficient heap memory

**"Validation failed for op"**
- This indicates a potential issue with an op mapping
- Use `--debug` flag for detailed information
- Review the specific error details in the output
- Check the original op definition in the registry files

**"PreImportHook not found"**
- Use debug mode to investigate: `--debug`
- Check that all dependencies are built and in classpath
- See `DEBUG_INVESTIGATION.md` for detailed steps

**"Permission denied"**
- Make script executable: `chmod +x update-op-registry.sh`
- Ensure write permissions to target resource directories

### Debug Mode
Use `--debug` flag for detailed output during build and execution phases, especially useful for investigating PreImportHook discovery issues.

## Comparison with Original Tool

| Aspect | Original (platform-tests) | Standalone (contrib) |
|--------|---------------------------|---------------------|
| Location | `platform-tests/` | `contrib/op-registry-updater/` |
| Dependencies | Full platform-tests + all test deps | Only framework import modules |
| Build time | ~5-10 minutes | ~1-2 minutes |
| Maintenance | Mixed with test code | Dedicated tool |
| Interface | Test-focused | Purpose-built CLI |
| Debug capabilities | Limited | Built-in PreImportHook debugging |

## Related Files

### Replaced Files:
- Original script: `<project-root>/update-op-registry.sh`
- Original class: `platform-tests/src/main/kotlin/org/eclipse/deeplearning4j/frameworkimport/runner/OpRegistryUpdater.kt`

### Registry Definition Files:
- TensorFlow: `nd4j/samediff-import/samediff-import-tensorflow/src/main/kotlin/org/nd4j/samediff/frameworkimport/tensorflow/definitions/TensorflowOpDeclarations.kt`
- ONNX: `nd4j/samediff-import/samediff-import-onnx/src/main/kotlin/org/nd4j/samediff/frameworkimport/onnx/definitions/OnnxOpDeclarations.kt`

### Test Files (no longer needed for updates):
- `platform-tests/src/test/kotlin/org/eclipse/deeplearning4j/frameworkimport/frameworkimport/tensorflow/loader/TestTensorflowProcessLoader.kt`
- `platform-tests/src/test/kotlin/org/eclipse/deeplearning4j/frameworkimport/frameworkimport/onnx/loader/TestOnnxProcessLoader.kt`
