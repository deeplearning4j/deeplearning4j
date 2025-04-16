# ADR 0035: SameDiff Unified Container Format

## Status

Proposed

Proposed by: Adam Gibson (15-04-2025)
Discussed with: Engineering Team

## Context

The current SameDiff serialization relies on FlatBuffers for graph representation and handles large arrays (>2GB) using a chunking mechanism. However, this approach has several limitations:

1. **Single File Deployment**: Current format often requires multiple files when externalizing large arrays
2. **Large Model Support**: Limited efficiency when dealing with very large models
3. **Metadata Management**: Lack of standardized metadata for model tracking and versioning
4. **Model Sharding**: Limited explicit support for sharding large models
5. **Compatibility**: Each format change risks breaking backward compatibility

We need a more robust serialization format that addresses these challenges while maintaining compatibility with existing systems.

## Decision

We will implement a unified container format for SameDiff that encapsulates both graph structure and arrays in a single file, with support for optional externalization and sharding when needed. This format will maintain full backward compatibility with the original serialization approach.

### Key Components

1. **Section-Based Container Structure**:
   - Header section with magic number and version
   - Metadata section for model information
   - Graph section containing FlatBuffers serialized graph
   - Arrays section containing all parameter arrays
   - Shard information section for sharded models

2. **Section Format**:
   - Section type identifier (1 byte)
   - Section length (8 bytes)
   - Optional compression flag
   - Section data
   - Optional CRC checksum

3. **Large Array Handling**:
   - In-container chunking for arrays of any size
   - Efficient storage with optional compression
   - Lazy loading capability for large arrays

4. **Metadata Management**:
   - Standardized keys for common model attributes
   - Support for custom metadata
   - Versioning information

5. **Sharding Support**:
   - Explicit first-class support for model sharding
   - Smart distribution of variables across shards
   - Shard reference system for model reconstruction

6. **Backward Compatibility**:
   - Automatic format detection between original and container formats
   - Support for loading both internal and externalized original formats
   - Legacy model conversion utilities

### Implementation Details

1. **Container Format**:
   ```
   MAGIC_BYTES (12 bytes: "SAMEDIFF_MODEL")
   VERSION (4 bytes)
   [SECTION_HEADER, length, data]
   [SECTION_METADATA, length, data]
   [SECTION_GRAPH, length, data]
   [SECTION_ARRAYS, length, data]
   [SECTION_SHARD_INFO, length, data] (optional)
   ```

2. **Compression Options**:
   - None (0): No compression
   - Deflate (1): Standard deflate compression
   - Configurable compression levels

3. **Backward Compatibility Approach**:
   - Format detection based on magic number/byte patterns
   - Automatic redirection to appropriate loaders
   - Handling of both standard and externalized legacy formats
   - Consideration of legacy shard naming patterns

4. **API Design**:
   ```java
   // Basic save/load - automatically handles both formats
   SameDiffContainerFormat.save(sameDiff, file, saveUpdaterState);
   SameDiff model = SameDiffContainerFormat.load(file, loadUpdaterState);
   
   // Advanced options
   SameDiffContainerFormat.save(sameDiff, file, saveUpdaterState, metadata, compression, compressionLevel);
   
   // Sharding
   SameDiffContainerFormat.saveSharded(sameDiff, baseFile, saveUpdaterState, numShards);
   SameDiff model = SameDiffContainerFormat.loadSharded(baseFile, loadUpdaterState);
   
   // Format conversion utilities
   SameDiffContainerFormat.convertLegacyToContainer(legacyFile, newContainerFile);
   ```

## Consequences

### Advantages

1. **Simplified Deployment**:
   - Single file deployment for most models
   - Easier distribution and management
   - Reduced risk of missing files

2. **Enhanced Model Storage**:
   - Support for models of any size
   - Efficient storage with compression
   - Selective loading of model components

3. **Better Metadata Management**:
   - Standardized tracking of model attributes
   - Version management for compatibility
   - Custom metadata for specific requirements

4. **First-Class Sharding**:
   - Explicit support for very large models
   - Intelligent variable distribution
   - Efficient loading of sharded models

5. **Complete Backward Compatibility**:
   - Seamless support for reading existing formats
   - Automatic format detection and handling
   - No disruption to existing workflows
   - Migration path for older models

### Disadvantages

1. **Implementation Complexity**:
   - More complex than current FlatBuffers-only approach
   - Need for comprehensive testing across formats
   - Additional code maintenance burden

2. **Performance Considerations**:
   - Potential overhead for small models
   - Compression/decompression time
   - Additional memory usage during serialization

3. **Adoption Effort**:
   - Updates needed to existing tooling
   - Documentation for new format features
   - Migration of existing models

## Technical Details

### Format Detection Algorithm
```
1. Read first 12 bytes of file
2. If bytes match SAMEDIFF_MODEL magic number:
   a. Parse as container format
3. Else if bytes match FlatBuffers header pattern:
   a. Parse as original FlatBuffers format
   b. Check for external arrays if referenced
4. Else:
   a. Throw unsupported format exception
```

### Original Format Support
```java
public static SameDiff load(File file, boolean loadUpdaterState) throws IOException {
    // Try to detect format
    byte[] header = readFileHeader(file, 12);
    
    if (Arrays.equals(header, CONTAINER_MAGIC_BYTES)) {
        // This is a container format
        return loadContainer(file, loadUpdaterState);
    } else {
        // Assume original format
        return SameDiffSerializer.load(file, loadUpdaterState);
    }
}
```

### External Array Handling
```java
private static SameDiff loadOriginalWithExternals(File file, boolean loadUpdaterState) {
    // Load main file
    SameDiff sd = SameDiffSerializer.load(file, loadUpdaterState);
    
    // Check for external files
    String baseName = file.getName().replaceFirst("[.][^.]+$", "");
    File parentDir = file.getParentFile();
    File externalDir = new File(parentDir, baseName + "_external");
    
    if (externalDir.exists()) {
        // Load external arrays and add to model
        loadExternalArrays(sd, externalDir);
    }
    
    return sd;
}
```

### Implementation Classes
- `SameDiffContainerFormat`: Main API class
- `ContainerWriter`: Handles writing to container format
- `ContainerReader`: Handles reading from container format
- `FormatDetector`: Detects and routes to appropriate loader
- `LegacyFormatHandler`: Handles original format specifics
- `ContainerConverter`: Converts between formats

## Alternatives Considered

1. **Enhanced FlatBuffers Only**:
   - Pros: Simpler implementation, builds on existing code
   - Cons: Limited ability to handle very large models, continuation of multi-file approach

2. **HDF5-Based Format**:
   - Pros: Mature library with chunking and compression
   - Cons: Additional dependency, complex Java integration

3. **Custom Binary Format Without Sections**:
   - Pros: Potentially simpler implementation
   - Cons: Less flexible, harder to extend, more brittle

4. **Protobuf with External Arrays**:
   - Pros: Simpler schema evolution
   - Cons: Same multi-file limitations as current approach

5. **Zip Archive of Components**:
   - Pros: Simple to implement, standard format
   - Cons: Less efficient, higher overhead, no streaming support

6. **New Format Without Legacy Support**:
   - Pros: Cleaner implementation, no legacy burden
   - Cons: Breaking change for users, migration difficulties