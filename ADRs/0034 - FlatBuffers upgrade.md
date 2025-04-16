# ADR 0034: FlatBuffers Modernization

## Status

Implemented

Proposed by: Assistant (14-04-2025)
Discussed with: Adam Gibson

## Context

The libnd4j library uses FlatBuffers for serialization of neural network graphs and related data structures. The current implementation uses FlatBuffers 1.12.0 syntax and conventions, particularly for handling sequences and arrays. With the upgrade to newer versions of FlatBuffers, we need to modernize our schema definitions and code generation.

The primary challenges include:

1. Migration from legacy Sequence/SequenceItem patterns to modern vector types
2. Proper generation of Java files with correct package structures
3. Integration with build system for consistent compilation
4. Maintaining backward compatibility where possible
5. Ensuring proper environment variable passing for code generation
6. Proper schema namespace management

## Decision

We implement a modernized FlatBuffers integration that uses current vector syntax and build processes. This includes:

### Key Components
- Schema files using modern vector syntax (`[Type]` instead of Sequence)
- Direct build process integration in CMake
- Explicit environment variable handling for flatc compiler
- Namespace standardization across schema files

### Implementation Details
1. Schemas use vector syntax for array types (e.g., `[FlatArray]` instead of `Sequence<FlatArray>`)
2. CMake executes flatc compilation directly instead of using custom targets
3. Build process ensures flatc is compiled before schema generation
4. Java package structure matches expected nd4j-api layout
5. Schema namespaces standardized to `graph` without `sd` prefix

### Schema Example
```fbs
namespace graph;

table SequenceItem {
  name:string;
  associated_variable:[FlatArray];
}

table SequenceItemRoot {
  sequence_items:[SequenceItem];
}

root_type SequenceItemRoot;
```

### Build Integration
```cmake
execute_process(
    COMMAND ${CMAKE_COMMAND} -E env "FLATC_PATH=${FLATC_EXECUTABLE}" 
    bash ${CMAKE_CURRENT_SOURCE_DIR}/flatc-generate.sh
    RESULT_VARIABLE FLATC_RESULT
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
```

## Consequences

### Advantages
1. Modern FlatBuffers Integration:
   - Cleaner schema definitions
   - Better type safety through vector syntax
   - More maintainable code generation
   - Consistent with current FlatBuffers best practices

2. Build System Integration:
   - Reliable flatc compilation
   - Proper environment variable handling
   - Direct process execution instead of custom targets
   - Better error handling and reporting

3. Code Organization:
   - Correct Java package structure
   - Standardized namespaces
   - Clear separation of generated code
   - Better integration with existing nd4j structure

### Disadvantages
1. Implementation Requirements:
   - Need to update existing schema files
   - Must ensure build process compatibility
   - Potential for temporary build issues during transition

2. Migration Effort:
   - Updates needed for existing code using legacy patterns
   - Need to verify all schema files
   - Testing required for all serialization paths

## Technical Details

### Schema Location
```
/libnd4j/include/graph/scheme/*.fbs
```

### Generated Code Location
```
/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/graph/
```

### Build Process
1. CMake configures build
2. flatc compiler is built
3. Schema generation script runs
4. Java files are copied to appropriate location

## Alternatives Considered

1. Custom Target Approach:
   - Pros: More traditional CMake integration
   - Cons: Less direct control over process, harder to debug

2. Manual File Generation:
   - Pros: Simpler build process
   - Cons: Error-prone, harder to maintain

3. Separate Build Step:
   - Pros: Cleaner separation of concerns
   - Cons: More complex build process, potential for synchronization issues