# ADR 0035: SameDiff Unified Container Format

## Status

Implemented

Proposed by: Adam Gibson (15-04-2025)  


## Context

The current SameDiff serialization relies on FlatBuffers for graph representation and handles large arrays (>2GB) using a chunking mechanism. However, this approach has several limitations:

1. **Single File Deployment**: Current format often requires multiple files when externalizing large arrays
2. **Large Model Support**: Limited efficiency when dealing with very large models
3. **Metadata Management**: Lack of standardized metadata for model tracking and versioning
4. **Model Sharding**: Limited explicit support for sharding large models
5. **Compatibility**: Each format change risks breaking backward compatibility

We need a more robust serialization format that addresses these challenges while maintaining compatibility with existing systems.

## Decision

We have implemented a unified container format for SameDiff that encapsulates both graph structure and arrays in a single file, with support for optional externalization and sharding when needed. This format maintains full backward compatibility with the original serialization approach.

### Key Components

1. **Multi-Format Support**:
   - SDNB Format: Single-file internal format (.sdnb)
   - SDZ Format: ZIP-based container format (.sdz)
   - Sharded formats for both SDNB and SDZ

2. **SDNB Format**:
   - Section-based container with header, metadata, graph, and arrays
   - Efficient memory mapping for large arrays
   - Optimized for performance with direct I/O

3. **SDZ Format**:
   - Standard ZIP archive containing internal .sdnb files
   - Compressed storage to reduce file size
   - Standard tools compatibility for inspection and extraction
   - Single file distribution for complex models

4. **Metadata Management**:
   - Standardized keys for common model attributes
   - Support for custom metadata
   - Versioning and provenance information

5. **Sharding Support**:
   - Explicit first-class support for model sharding in both formats
   - Smart distribution of variables across shards
   - Automatic shard count determination based on model size
   - Consistent naming convention for shards

6. **Backward Compatibility**:
   - Automatic format detection between SDNB and SDZ formats
   - Support for loading both internal and externalized original formats
   - Legacy model conversion utilities

### Implementation Details

1. **SDNB Format Structure**:
   ```
   MAGIC_BYTES (4 bytes: "SDNB")
   VERSION (4 bytes)
   MANIFEST_OFFSET (8 bytes)
   MANIFEST_LENGTH (8 bytes)
   METADATA_OFFSET (8 bytes)
   [FLATBUFFER_GRAPH_DATA]
   [APPENDED_ARRAYS_DATA]
   [SERIALIZED_MANIFEST]
   ```

2. **SDZ Format Structure**:
   ```
   ZIP_HEADER
   [ENTRY: model.sdnb]           # Graph structure shard
   [ENTRY: model.shard0-of-N.sdnb] # Alternative naming for graph shard
   [ENTRY: model.shard1-of-N.sdnb] # Variable shard 1
   [ENTRY: model.shard2-of-N.sdnb] # Variable shard 2
   ...
   [ENTRY: model.shardM-of-N.sdnb] # Variable shard M
   ZIP_DIRECTORY
   ZIP_END
   ```

3. **Sharding Strategy**:
   - Graph structure in shard 0
   - Variables distributed across remaining shards
   - Dynamic shard count calculation based on variable sizes
   - Maximum shard size limit of 1GB per shard
   - Smart variable grouping to minimize cross-shard dependencies

4. **API Design**:
   ```java
   // SDNB Format API
   SameDiffSerializer.save(sameDiff, file, saveUpdaterState, metadata);
   SameDiffSerializer.saveAutoShard(sameDiff, baseFile, saveUpdaterState, metadata);
   SameDiffSerializer.saveSharded(sameDiff, baseFile, saveUpdaterState, estimatedShards, metadata);
   SameDiff model = SameDiffSerializer.load(file, loadUpdaterState);
   SameDiff model = SameDiffSerializer.loadSharded(baseFile, loadUpdaterState);
   
   // SDZ Format API
   SDZSerializer.save(sameDiff, outputZipFile, saveUpdaterState, metadata);
   SameDiff model = SDZSerializer.load(modelZipFile, loadUpdaterState);
   ```

## Implementation

### SDZ Format Details

The SDZ format addresses the need for single-file distribution of large models through the following implementation:

1. **ZIP Container**: The SDZ format uses a standard ZIP archive as its container, enabling compatibility with standard zip tools for inspection and extraction.

2. **Internal Structure**:
   - The ZIP archive contains one or more SDNB format files
   - The first file (shard0) contains the graph structure
   - Subsequent files contain variables distributed across shards
   - Consistent naming convention ensures proper loading sequence

3. **Sharding Implementation**:
   - `SDZSerializer.save()` internally calls `SameDiffSerializer.saveAutoShard()` to create SDNB files
   - These files are then compressed and packaged into the ZIP archive
   - Automatic cleanup of temporary files after ZIP creation
   - Distributed variable serialization across shards based on size

4. **Loading Process**:
   - `SDZSerializer.load()` extracts all SDNB files to a temporary directory
   - Loads shard 0 first to establish graph structure
   - Loads variable data from remaining shards
   - Ensures temporary directory cleanup
   - Returns fully reconstituted SameDiff instance

5. **ZIP Operations**:
   - Uses standard Java ZIP APIs for maximum compatibility
   - Implements efficient I/O with buffering for large file handling
   - Security measures against zip slip vulnerabilities
   - Validation of ZIP structure integrity

6. **Optimizations**:
   - Manifest-based array lookup for efficient loading
   - Smart buffer management to minimize memory pressure
   - Native byte order handling for cross-platform compatibility
   - Verification steps to validate loaded model integrity

### Performance Considerations

The SDZ format balances compression benefits against performance requirements:

1. **Serialization Performance**:
   - Slight additional overhead for ZIP compression
   - Parallelized compression when possible
   - Progressive ZIP writing to avoid memory spikes

2. **Deserialization Performance**:
   - Sequential extraction for predictable memory usage
   - Lazy loading strategies for large variables
   - Efficient memory mapping for large arrays when possible
   - Verification during loading to ensure data integrity

3. **Storage Efficiency**:
   - Typically 30-50% size reduction through compression
   - Optimal balance between compression level and performance
   - Compression ratio varies based on parameter data patterns

## Consequences

### Advantages

1. **Simplified Deployment**:
   - Single file deployment with SDZ format
   - Easier distribution and management
   - Reduced risk of missing files or shard mismatches

2. **Enhanced Model Storage**:
   - Support for models of any size
   - Efficient storage with ZIP compression
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
   - More complex than previous FlatBuffers-only approach
   - Additional code paths for format handling
   - Need for comprehensive testing across formats

2. **Performance Considerations**:
   - Compression/decompression time with SDZ format
   - Temporary storage requirements during extraction
   - Slight overhead for small models

3. **Tool Ecosystem**:
   - Need for updates to existing tooling
   - Additional format documentation requirements
   - Migration guidance for existing models

## Technical Implementation

### Format Detection Algorithm
```java
public static SameDiff load(File file, boolean loadUpdaterState) throws IOException {
    // Check if it's a ZIP file first (SDZ format)
    if (isZipFile(file)) {
        return SDZSerializer.load(file, loadUpdaterState);
    }
    
    // Not a ZIP, check if it's a native SDNB file
    if (isValidSdnbFile(file)) {
        return SameDiffSerializer.load(file, loadUpdaterState);
    }
    
    // Check if it's a base name for sharded files
    if (hasShardedFiles(file)) {
        return SameDiffSerializer.loadSharded(file, loadUpdaterState);
    }
    
    // Unsupported format
    throw new UnsupportedOperationException("Unrecognized model format");
}
```

### SDZ Implementation
```java
public static void save(SameDiff sameDiff, File outputZipFile, boolean saveUpdaterState, 
                        Map<String, String> metadata) throws IOException {
    // Create temporary directory for SDNB files
    Path tempDir = Files.createTempDirectory("sdz-serializer-save-");
    
    try {
        // Save using SDNB serializer to temporary directory
        File internalSavePath = new File(tempDir.toFile(), "model");
        SameDiffSerializer.saveAutoShard(sameDiff, internalSavePath, saveUpdaterState, metadata);
        
        // Collect all files to add to ZIP
        List<File> filesToZip = new ArrayList<>();
        findAllFilesRecursively(tempDir.toFile(), filesToZip);
        
        // Create ZIP archive
        createZipArchive(outputZipFile, filesToZip);
    } finally {
        // Clean up temporary directory
        FileUtils.deleteDirectory(tempDir.toFile());
    }
}

public static SameDiff load(File modelZipFile, boolean loadUpdaterState) throws IOException {
    // Extract ZIP to temporary directory
    Path tempDir = Files.createTempDirectory("sdz-serializer-load-");
    
    try {
        // Extract ZIP contents
        extractZip(modelZipFile, tempDir.toFile());
        
        // Determine the path to load from
        File loadPath = determineLoadPath(tempDir.toFile());
        
        // Load using SDNB serializer
        return SameDiffSerializer.load(loadPath, loadUpdaterState);
    } finally {
        // Clean up temporary directory
        FileUtils.deleteDirectory(tempDir.toFile());
    }
}
```


## Migration Guidelines

For existing users:

1. **Loading Existing Models**:
   - No changes needed, automatic format detection handles existing models

2. **Converting to SDZ Format**:
   - Use `SDZSerializer.save()` with existing SameDiff instances
   - Alternatively, load existing models and save in SDZ format

3. **When to Use Each Format**:
   - SDNB: For highest performance, particularly during training
   - SDZ: For deployment, storage efficiency, and single-file distribution
   - Sharded formats: For very large models exceeding memory limits