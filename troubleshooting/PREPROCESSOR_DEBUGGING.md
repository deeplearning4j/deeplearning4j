# Using libnd4j.preprocess for Macro Debugging

## Overview

The `libnd4j.preprocess` flag is a specialized debugging tool that runs only the C++ preprocessor, outputting the preprocessed source files before actual compilation. This is invaluable for:
- Debugging complex macro expansions
- Verifying include paths and header dependencies
- Understanding template instantiations
- Investigating conditional compilation issues

## Usage

### Basic Command
```bash
mvn clean install -Dlibnd4j.preprocess=ON
```

### Output Location
Preprocessed files are created in the `preprocessed` directory with the following naming convention:
- Directory structure is flattened with underscores
- Extension changes to `.i`
- Example: `include/ops/declarable/generic/transforms/reverse.cpp` becomes `preprocessed/ops_declarable_generic_transforms_reverse.i`

## What the Preprocessor Does

The preprocessor performs several transformations:
1. Macro expansion
2. Header file inclusion
3. Conditional compilation evaluation (#ifdef, #ifndef, etc.)
4. Line number and file name tracking
5. Comment removal
6. Macro constant substitution

## Implementation Details

From the CMakeLists.txt configuration:

1. Directory Setup:
```cmake
set(PREPROCESSED_DIR "${CMAKE_SOURCE_DIR}/preprocessed")
file(MAKE_DIRECTORY ${PREPROCESSED_DIR})
```

2. File Processing:
```cmake
# Command format used internally
${compiler} -E ${include_dirs} "${src}" -o "${preprocessed_file}"
```

3. Language Detection:
```cmake
if(src MATCHES "\\.c$")
    set(language "C")
elseif(src MATCHES "\\.cpp$|\\.cxx$|\\.cc$")
    set(language "CXX")
else()
    set(language "CXX")
endif()
```

## Common Use Cases

### 1. Debugging Complex Macros
When dealing with complex macro definitions like BUILD_DOUBLE_TEMPLATE:
```cpp
#if defined(SD_COMMON_TYPES_GEN) && defined(SD_COMMON_TYPES_@FL_TYPE_INDEX@)
BUILD_DOUBLE_TEMPLATE(template void someFunc, (arg_list,..),
                     SD_COMMON_TYPES_@FL_TYPE_INDEX@, SD_INDEXING_TYPES);
#endif
```

Using preprocessor output helps verify:
- Macro expansion correctness
- Template instantiation
- Type substitutions

### 2. Include Path Issues
When headers aren't found or wrong versions are included:
```bash
# Check full include path resolution
grep "#include" preprocessed/your_file.i
```

### 3. Conditional Compilation
For code with multiple #ifdef paths:
```cpp
#ifdef SD_CUDA
    // CUDA-specific code
#else
    // CPU code
#endif
```
The preprocessed output shows exactly which path was taken.

## Best Practices

1. **Organized Investigation**:
   ```bash
   # Create a directory for the specific issue
   mkdir debug_issue_123
   cd debug_issue_123
   
   # Run preprocessor and copy relevant files
   mvn clean install -Dlibnd4j.preprocess=ON
   cp ../preprocessed/relevant_file.i .
   ```

2. **Diff Comparison**:
   ```bash
   # Compare different configurations
   mv preprocessed/file.i file_config1.i
   # Change configuration
   mvn clean install -Dlibnd4j.preprocess=ON
   diff file_config1.i preprocessed/file.i
   ```

3. **Macro Tracing**:
   - Keep the original source file open
   - Search for specific macros in the preprocessed output
   - Use line directives (#line) to map back to source

## Troubleshooting Common Issues

1. **Missing Definitions**
   - Problem: Macros not expanding as expected
   - Solution: Check preprocessed output for undefined macros
   ```bash
   grep -n "undefined" preprocessed/*.i
   ```

2. **Include Order Issues**
   - Problem: Header dependencies not resolving correctly
   - Solution: Examine the order of #include directives in preprocessed output
   ```bash
   grep -n "#include" preprocessed/your_file.i
   ```

3. **Template Instantiation Problems**
   - Problem: Templates not generating expected code
   - Solution: Search for template instantiations in preprocessed output
   ```bash
   grep -n "template" preprocessed/your_file.i
   ```

## Tips for Large Codebases

1. **Filtering Output**:
   ```bash
   # Remove empty lines and comments
   grep -v '^$' preprocessed/file.i | grep -v '^//'
   
   # Focus on specific sections
   sed -n '/START_SECTION/,/END_SECTION/p' preprocessed/file.i
   ```

2. **Finding Macro Expansions**:
   ```bash
   # Search for specific macro expansions
   grep -A 5 -B 5 "MACRO_NAME" preprocessed/file.i
   ```

3. **Managing Output Size**:
   ```bash
   # Split large files for easier analysis
   split -l 1000 preprocessed/large_file.i split_
   
   # Create focused preprocessor outputs
   mv preprocessed better_name_preprocessed_$(date +%Y%m%d)
   ```